"""Quality probe: simulate int8 checkpoint storage by int8-round-tripping the
linear-attn (mamba) temporal state inside MambaPool.copy_from — the path that
loads a radix-cached state into a request on a prefix-cache HIT (and stores a
request state into the radix). Gated by a runtime FILE flag so one server run can
A/B: absent = bf16 baseline, present(/tmp/m2_int8_ckpt_on) = int8.

Per (head, last-channel) symmetric int8, matching the offline probe (~0.5% decode
error). Idempotent; backs up to .int8bak.
"""
import shutil
import sys

f = sys.argv[1] if len(sys.argv) > 1 else (
    "/sgl-workspace/sglang-theta2/python/sglang/srt/mem_cache/memory_pool.py"
)
src = open(f).read()
if "m2_int8_ckpt_on" in src:
    print("already patched")
    raise SystemExit(0)

# ensure `import os`
if "\nimport os\n" not in src:
    src = src.replace("\nimport torch\n", "\nimport os\nimport torch\n", 1)
    assert "\nimport os\n" in src, "failed to add import os"

anchor = (
    "        self.mamba_cache.temporal[:, dst_indices] = self.mamba_cache.temporal[\n"
    "            :, src_indices\n"
    "        ]"
)
assert anchor in src, "copy_from temporal anchor not found"
repl = (
    "        _t_src = self.mamba_cache.temporal[:, src_indices]\n"
    "        if os.path.exists(\"/tmp/m2_int8_ckpt_on\"):  # M2 int8-checkpoint quality probe\n"
    "            _amax = _t_src.abs().amax(dim=-2, keepdim=True).clamp(min=1e-8)\n"
    "            _sc = _amax / 127.0\n"
    "            _t_src = (\n"
    "                torch.round(_t_src / _sc).clamp(-127, 127) * _sc\n"
    "            ).to(self.mamba_cache.temporal.dtype)\n"
    "        self.mamba_cache.temporal[:, dst_indices] = _t_src"
)
src = src.replace(anchor, repl, 1)
shutil.copy(f, f + ".int8bak")
open(f, "w").write(src)
print("patched OK ->", f)
