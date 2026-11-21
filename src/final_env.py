"""Record the software and hardware used for the final-project runs (results/final_environment.md)."""
import os
import platform
import subprocess
import sys

import numpy
import pandas
import sklearn

import final_lib as fl
from common import RES

rows = [("python", sys.version.split()[0]), ("numpy", numpy.__version__), ("pandas", pandas.__version__),
        ("scikit-learn", sklearn.__version__), ("platform", platform.platform()), ("cpu cores", os.cpu_count())]
try:
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2 ** 30
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    rows += [("memory (GB)", f"{mem:.0f}"), ("cpu", chip)]
except Exception:  # non-macOS
    pass
try:
    import gurobipy
    rows.append(("gurobi", ".".join(map(str, gurobipy.gurobi.version()))))
except ImportError:
    rows.append(("gurobi", "not installed"))
rows.append(("IAI", "system image v3.2.2 (Julia 1.11.9)" if fl.iai_available() else "not available"))
rows.append(("seeds", "numpy/sklearn 0; IAI random_seed 1; Gurobi Seed 0"))
(RES / "final_environment.md").write_text("| item | value |\n|---|---|\n" + "".join(f"| {k} | {v} |\n" for k, v in rows))
print(open(RES / "final_environment.md").read())
