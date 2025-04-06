from __future__ import annotations

import asyncio
import logging
import struct
import threading
from functools import cache
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import zmq
from aiohttp import web

from sglang.srt.disaggregation.transfer_engine.mooncake import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode

logger = logging.getLogger(__name__)


class KVArgs:
    engine_rank: int
    tp_size: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


RequestPoolType = Dict[int, Tuple[npt.NDArray[np.int32], Optional[int]]]
WaitingPoolType = Dict[
    int, Tuple[str, list[int], npt.NDArray[np.int32], list[int], int]
]
KVSENDER_POLLING_PORT = 17788
KVRECEIVER_POLLING_PORT = 27788


class KVManager:
    # TODO: make it general and support multiple transfer backend before merging
    def __init__(self, args: KVArgs, disaggregation_mode: DisaggregationMode):
        self.engine = MooncakeTransferEngine()
        self.kv_args = args
        self.disaggregation_mode = disaggregation_mode
        self.request_pool: RequestPoolType = {}
        self.request_status: Dict[int, KVPoll] = {}
        self.server_socket = zmq.Context().socket(zmq.PULL)
        self.register_buffer_to_engine()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.waiting_pool: WaitingPoolType = {}
            self.transfer_event = threading.Event()
            self.start_prefill_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.start_decode_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def register_buffer_to_engine(self):
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            self.engine.register(kv_data_ptr, kv_data_len)

        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            self.engine.register(aux_data_ptr, aux_data_len)

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
    ):
        layer_num = int(len(self.kv_args.kv_data_ptrs) / 2)
        for layer_id in range(layer_num):
            prefill_key_layer_ptr = self.kv_args.kv_data_ptrs[layer_id]
            key_item_len = self.kv_args.kv_item_lens[layer_id]
            prefill_value_layer_ptr = self.kv_args.kv_data_ptrs[layer_num + layer_id]
            value_item_len = self.kv_args.kv_item_lens[layer_num + layer_id]

            decode_key_layer_ptr = dst_ptrs[layer_id]
            decode_value_layer_ptr = dst_ptrs[layer_num + layer_id]
            # TODO: Maybe combine multiple contiguous indices into one transfer_sync op
            for prefill_index, decode_index in zip(prefill_kv_indices, dst_kv_indices):
                prefill_key_addr = prefill_key_layer_ptr + prefill_index * key_item_len
                decode_key_addr = decode_key_layer_ptr + decode_index * key_item_len
                # TODO: mooncake transfer engine can do async transfer. Do async later
                status = self.engine.transfer_sync(
                    mooncake_session_id, prefill_key_addr, decode_key_addr, key_item_len
                )
                if status != 0:
                    return status

                prefill_value_addr = (
                    prefill_value_layer_ptr + prefill_index * value_item_len
                )
                decode_value_addr = (
                    decode_value_layer_ptr + decode_index * value_item_len
                )
                # TODO: mooncake transfer engine can do async transfer. Do async later
                status = self.engine.transfer_sync(
                    mooncake_session_id,
                    prefill_value_addr,
                    decode_value_addr,
                    value_item_len,
                )
                if status != 0:
                    return status
        return 0

    def send_aux(
        self,
        mooncake_session_id: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
    ):
        aux_item_len = self.kv_args.aux_item_lens[0]
        prefill_aux_addr = (
            self.kv_args.aux_data_ptrs[0] + prefill_aux_index * aux_item_len
        )
        decode_aux_addr = dst_aux_ptrs[0] + dst_aux_index * aux_item_len
        # TODO: mooncake transfer engine can do async transfer. Do async later
        # Not sure about the amount of aux data, maybe transfer it by zmq is more effective
        status = self.engine.transfer_sync(
            mooncake_session_id, prefill_aux_addr, decode_aux_addr, aux_item_len
        )
        return status

    def sync_status_to_decode_endpoint(self, remote: str, room: int):
        if ":" in remote:
            remote = remote.split(":")[0]
        // TODO(yuan-luo): support multiple decode on one node
        self._connect(
            "tcp://"
            + remote
            + ":"
            + str(KVRECEIVER_POLLING_PORT + self.kv_args.engine_rank)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(self.request_status[room]).encode("ascii"),
            ]
        )

    def start_prefill_thread(self):
        sender_rank_port = KVSENDER_POLLING_PORT + self.kv_args.engine_rank
        self.server_socket.bind("tcp://*:" + str(sender_rank_port))

        def prefill_thread():
            while True:
                (
                    endpoint,
                    mooncake_session_id,
                    bootstrap_room,
                    dst_ptrs,
                    dst_kv_indices,
                    dst_aux_ptrs,
                    dst_aux_index,
                ) = self.server_socket.recv_multipart()
                if bootstrap_room.decode("ascii") == "None":
                    continue
                endpoint = endpoint.decode("ascii")
                mooncake_session_id = mooncake_session_id.decode("ascii")
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                dst_ptrs = list(struct.unpack(f"{len(dst_ptrs)//8}q", dst_ptrs))
                dst_kv_indices = np.frombuffer(dst_kv_indices, dtype=np.int32)
                dst_aux_ptrs = list(
                    struct.unpack(f"{len(dst_aux_ptrs)//8}q", dst_aux_ptrs)
                )
                dst_aux_index = int(dst_aux_index.decode("ascii"))
                self.waiting_pool[bootstrap_room] = (
                    endpoint,
                    mooncake_session_id,
                    dst_ptrs,
                    dst_kv_indices,
                    dst_aux_ptrs,
                    dst_aux_index,
                )
                self.transfer_event.set()

        threading.Thread(target=prefill_thread).start()

        def transfer_thread():
            while True:
                self.transfer_event.wait()
                self.transfer_event.clear()
                bootstrap_room_ready = self.request_pool.keys()
                bootstrap_room_request = self.waiting_pool.keys()
                for room in list(bootstrap_room_request):
                    if room not in list(bootstrap_room_ready):
                        continue
                    status = KVPoll.Transferring
                    self.request_status[room] = status
                    (
                        endpoint,
                        mooncake_session_id,
                        dst_ptrs,
                        dst_kv_indices,
                        dst_aux_ptrs,
                        dst_aux_index,
                    ) = self.waiting_pool.pop(room)
                    self.sync_status_to_decode_endpoint(endpoint, room)
                    (
                        prefill_kv_indices,
                        prefill_aux_index,
                    ) = self.request_pool.pop(room)
                    ret = self.send_kvcache(
                        mooncake_session_id,
                        prefill_kv_indices,
                        dst_ptrs,
                        dst_kv_indices,
                    )
                    if ret != 0:
                        status = KVPoll.Failed
                        self.sync_status_to_decode_endpoint(endpoint, room)
                        continue
                    ret = self.send_aux(
                        mooncake_session_id,
                        prefill_aux_index,
                        dst_aux_ptrs,
                        dst_aux_index,
                    )
                    if ret != 0:
                        status = KVPoll.Failed
                    else:
                        status = KVPoll.Success
                    self.request_status[room] = status
                    self.sync_status_to_decode_endpoint(endpoint, room)

        threading.Thread(target=transfer_thread).start()

    def start_decode_thread(self):
        receiver_rank_port = KVRECEIVER_POLLING_PORT + self.kv_args.engine_rank
        self.server_socket.bind("tcp://*:" + str(receiver_rank_port))

        def decode_thread():
            while True:
                (bootstrap_room, status) = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                self.request_status[bootstrap_room] = status

        threading.Thread(target=decode_thread).start()

    def enqueue_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int],
    ):
        self.request_pool[bootstrap_room] = (kv_indices, aux_index)
        self.request_status[bootstrap_room] = KVPoll.WaitingForInput
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_event.set()

    def check_status(self, bootstrap_room: int):
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.request_status[bootstrap_room] == KVPoll.Success
        ):
            if bootstrap_room in self.request_pool:
                self.request_pool.pop(bootstrap_room)

        return self.request_status[bootstrap_room]

    def set_status(self, bootstrap_room: int, status: KVPoll):
        self.request_status[bootstrap_room] = status

    def get_localhost(self):
        return self.engine.get_localhost()

    def get_session_id(self):
        return self.engine.get_session_id()


class KVSender:

    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.kv_mgr.set_status(bootstrap_room, KVPoll.WaitingForInput)
        self.aux_index = None
        self.bootstrap_server_url = bootstrap_addr

    @cache
    def _connect_router(self, endpoint: str):
        socket = zmq.Context().socket(zmq.DEALER)
        self.identity = str(uuid.uuid4()).encode()
        socket.setsockopt(zmq.IDENTITY, self.identity)
        socket.connect(endpoint)
        return socket

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.aux_index = aux_index
        self.num_kv_indices = num_kv_indices
        self._connect_router("tcp://" + self.bootstrap_server_url).send_multipart(
            [
                "Prefill".encode("ascii"),
                str(self.kv_mgr.engine_rank).encode("ascii"),
                str(self.kv_mgr.tp_size).encode("ascii"),
            ]
        )

    def send(self, kv_indices: npt.NDArray[np.int32]):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, self.aux_index)

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:

    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None,
        prefill_addr: str,
    ):
        self.bootstrap_room = bootstrap_room
        self.bootstrap_addr = bootstrap_addr
        self.prefill_addr = prefill_addr
        self.kv_mgr = mgr
        self.decode_ip = self.kv_mgr.get_localhost()
        self.session_id = self.kv_mgr.get_session_id()
        self.kv_mgr.set_status(bootstrap_room, KVPoll.WaitingForInput)
        self.prefill_engine_rank = None
        self.prefill_tp_size = None

    @cache
    def _connect_router(self, endpoint: str):
        socket = zmq.Context().socket(zmq.DEALER)
        self.identity = str(uuid.uuid4()).encode()
        socket.setsockopt(zmq.IDENTITY, self.identity)
        socket.connect(endpoint)
        return socket

    @cache
    def _connect(self, endpoint: str):
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(endpoint)
        return socket

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        # Send to bootstrap server
        self._connect("tcp://" + self.bootstrap_addr).send_multipart(
            [
                "Decode".encode("ascii"),
                str(0).encode("ascii"),
                str(0).encode("ascii"),
            ]
        )
        # Start listen clients thread
        self.zmq_thread = threading.Thread(target=self._listen_server, args=(kv_indices, aux_index))
        self.zmq_thread.start()
        self.zmq_thread.join()

    def _listen_server(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        # Receive messages from bootstrap server (Prefill)
        (role, engine_rank, tp_size) = self.router_socket.recv_multipart()
        role = role.decode("ascii")
        if role == "Decode":
            self.prefill_engine_rank = int(engine_rank.decode("ascii"))
            self.prefill_tp_size = int(tp_size.decode("ascii"))
            self.handshake_prefill_server(kv_indices, aux_index)

    def handshake_prefill_server(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        self.kv_mgr.enqueue_request(self.bootstrap_room, kv_indices, aux_index)
        packed_kv_data_ptrs = b"".join(
            struct.pack("q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )

        sender_polling_port = int(self.prefill_addr.split(":")[1])
        self.prefill_server_url = (
            self.bootstrap_addr.split(":")[0]
            + ":"
            + str(sender_polling_port + self.prefill_engine_rank)
        )
        self._connect("tcp://" + self.prefill_server_url).send_multipart(
            [
                self.decode_ip.encode("ascii"),
                self.session_id.encode("ascii"),
                str(self.bootstrap_room).encode("ascii"),
                packed_kv_data_ptrs,
                kv_indices.tobytes(),
                packed_aux_data_ptrs,
                str(aux_index).encode("ascii"),
            ]
        )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int):
        self.route_port = port
        self.app = web.Application()
        self.store = dict()
        self.lock = asyncio.Lock()
        self._setup_routes()

        self.context = zmq.Context()

        # ROUTER socket to communicate with Prefill and Decode
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://*:{self.route_port}")

        # TODO(luo-yuan): register decode information
        self.prefill_engine_rank = None
        self.prefill_tp_size = None

        # Start bootstrap server
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

        # Start listen clients thread
        self.zmq_thread = threading.Thread(target=self._listen_clients, daemon=True)  # ZMQ communication thread
        self.zmq_thread.start()

    def _listen_clients(self):
        while True:
            # Receive messages from clients (Prefill or Decode)
            (role, engine_rank, tp_size) = self.router_socket.recv_multipart()
            role = role.decode("ascii")
            engine_rank = int(engine_rank.decode("ascii"))
            tp_size = int(tp_size.decode("ascii"))
            if role == "Prefill":
                self._handle_prefill(engine_rank, tp_size)
            elif role == "Decode":
                self._handle_decode()

    def _handle_prefill(self, engine_rank, tp_size):
        """Handle Prefill message"""
        self.prefill_engine_rank = engine_rank
        self.prefill_tp_size = tp_size

    def _handle_decode(self):
        """Handle Decode message"""
        if self.prefill_engine_rank is None or self.prefill_tp_size is None:
            print("Error: Metadata not yet received from Prefill.")
            self.router_socket.send_multipart([identity, b"Error: Metadata not ready"])
        else:
            self.router_socket.send_multipart(
                [
                    "Decode".encode("ascii"),
                    str(prefill_engine_rank).encode("ascii"),
                    str(prefill_tp_size).encode("ascii"),
                ]
            )

    def _setup_routes(self):
        self.app.router.add_route("*", "/metadata", self._handle_metadata)

    async def _handle_metadata(self, request: web.Request):
        key = request.query.get("key", "")

        if request.method == "GET":
            return await self._handle_get(key)
        elif request.method == "PUT":
            return await self._handle_put(key, request)
        elif request.method == "DELETE":
            return await self._handle_delete(key)
        return web.Response(
            text="Method not allowed", status=405, content_type="application/json"
        )

    async def _handle_get(self, key):
        async with self.lock:
            value = self.store.get(key)
        if value is None:
            return web.Response(
                text="metadata not found", status=404, content_type="application/json"
            )
        return web.Response(body=value, status=200, content_type="application/json")

    async def _handle_put(self, key, request):
        data = await request.read()
        async with self.lock:
            self.store[key] = data
        return web.Response(
            text="metadata updated", status=200, content_type="application/json"
        )

    async def _handle_delete(self, key):
        async with self.lock:
            if key not in self.store:
                return web.Response(
                    text="metadata not found",
                    status=404,
                    content_type="application/json",
                )
            del self.store[key]
        return web.Response(
            text="metadata deleted", status=200, content_type="application/json"
        )

    def _run_server(self):
        try:
            # Event Loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            self._runner = web.AppRunner(self.app)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, port=self.port)
            self._loop.run_until_complete(site.start())
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            # Cleanup
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()

    def close(self):
        """Shutdown"""
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("Stopping server loop...")

        if self.thread.is_alive():
            self.thread.join(timeout=2)
            logger.info("Server thread stopped")

    def poll(self) -> KVPoll: ...