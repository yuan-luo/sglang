"""M0 instrumentation: count full-token-match vs mamba-token-match in
MambaRadixCache._match_prefix_helper. The gap = prefix whose KV is cached but
whose SSM state is missing -> forced recompute (exactly what B would eliminate).

Writes cumulative stats to /tmp/m0_stats.json (scheduler-process logger.info is
NOT captured in the server log file, so we dump to a file instead). Also records
the mamba pool size (S_m) and a "severe gap" counter (mamba_match < 50% of
full_match -> eviction, vs small gap = chunk alignment).

Idempotent; backs up to .m0bak. Run on the box with the target repo path arg.
"""
import shutil
import sys

f = sys.argv[1] if len(sys.argv) > 1 else (
    "/sgl-workspace/sglang-theta2/python/sglang/srt/mem_cache/mamba_radix_cache.py"
)
src = open(f).read()
if "[M0STATS]" in src:
    print("already patched")
    raise SystemExit(0)

glob_anchor = "logger = logging.getLogger(__name__)"
assert glob_anchor in src, "logger anchor not found"
src = src.replace(
    glob_anchor,
    glob_anchor
    + '\n\nimport json as _json\n'
    + '_M0 = {"n": 0, "full": 0, "mamba": 0, "recompute": 0, "n_gap": 0, '
    '"n_severe": 0, "smax": 0}',
    1,
)

anchor = "        return value, best_last_node, best_value_len"
assert anchor in src, "return anchor not found"
inject = (
    "        try:\n"
    "            _ft = sum(len(v) for v in value)\n"
    "            _mt = sum(len(v) for v in value[:best_value_len])\n"
    "            if _ft > 0:\n"
    '                _M0["n"] += 1\n'
    '                _M0["full"] += _ft\n'
    '                _M0["mamba"] += _mt\n'
    '                _M0["recompute"] += (_ft - _mt)\n'
    "                if _ft > _mt:\n"
    '                    _M0["n_gap"] += 1\n'
    "                if _mt < 0.5 * _ft:\n"
    '                    _M0["n_severe"] += 1\n'
    '                if _M0["smax"] == 0:\n'
    "                    try:\n"
    '                        _M0["smax"] = int(self.req_to_token_pool.mamba_pool.size)\n'
    "                    except Exception:\n"
    "                        pass\n"
    '                if _M0["n"] % 100 == 0:\n'
    '                    _M0["waste_frac"] = _M0["recompute"] / max(1, _M0["full"])\n'
    "                    try:\n"
    '                        with open("/tmp/m0_stats.json", "w") as _mf:\n'
    "                            _json.dump(_M0, _mf)\n"
    "                    except Exception:\n"
    "                        pass\n"
    "        except Exception:\n"
    "            pass\n"
)
src = src.replace(anchor, inject + anchor, 1)
shutil.copy(f, f + ".m0bak2")
open(f, "w").write(src)
print("patched OK ->", f)
