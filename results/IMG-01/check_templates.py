"""IMG-01 evidence: compile and run every graph template of a pwm_core tree and hash the forward output.

    PWM_CORE_SRC=<.../packages/pwm_core> PYTHONPATH=. OMP_NUM_THREADS=1 nice -n 19 python results/IMG-01/check_templates.py OUT.json

For each template: does it compile and run on rng(0).random(x_shape), which aliases/repairs this repo had to apply,
and the sha256 of the forward output (noise is seeded, so it is deterministic). Run on the unpatched pwm_core
(3899e328) and on the same tree with results/IMG-01/pwm_core_templates.patch applied, then diff the two reports.
"""
import hashlib
import json
import signal
import sys
from pathlib import Path

import numpy as np

from pwm_mode.imaging import core

out = Path(sys.argv[1])
templates = core.contrib("graph_templates.yaml")["templates"]
src = core.PWM_CORE_SRC / "contrib" / "graph_templates.yaml"
rows = {}
for tid, t in sorted(templates.items()):
    nodes, notes = core.normalise_nodes(t["nodes"])
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("timeout")))
    signal.alarm(15)
    try:
        op = core.compile_graph(nodes, t["edges"], t.get("metadata", {}))
        x = np.random.default_rng(0).random(tuple(t.get("metadata", {}).get("x_shape", [64, 64])))
        y = np.asarray(op.forward(x))
        rows[tid] = {"runs": bool(np.all(np.isfinite(y.astype(complex)))), "dtype": str(y.dtype), "shape": list(y.shape),
                     "aliases_applied": notes, "repaired_by": t.get("metadata", {}).get("repaired_by"),
                     "forward_sha256": hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()}
    except Exception as e:  # noqa: BLE001 - the error text is the record
        rows[tid] = {"runs": False, "error": f"{type(e).__name__}: {str(e)[:160]}", "aliases_applied": notes}
    finally:
        signal.alarm(0)
rep = {"graph_templates_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "n_templates": len(rows),
       "n_run": sum(r["runs"] for r in rows.values()), "failing": sorted(k for k, r in rows.items() if not r["runs"]),
       "n_with_alias": sum(bool(r["aliases_applied"]) for r in rows.values()),
       "n_repaired": sum(bool(r.get("repaired_by")) for r in rows.values()), "templates": rows}
out.write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "templates"}, indent=1))
