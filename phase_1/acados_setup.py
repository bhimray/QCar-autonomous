import sys
import os

ACADOS_DIR = r"C:\Users\rayyu\acados"

# Python interface (acados_template)
sys.path.insert(0, os.path.join(
    ACADOS_DIR, "interfaces", "acados_template"
))

# Windows DLL path (THIS MUST MATCH YOUR BUILD OUTPUT)
if os.name == "nt":
    os.add_dll_directory(r"C:\Users\rayyu\acados\bin")
