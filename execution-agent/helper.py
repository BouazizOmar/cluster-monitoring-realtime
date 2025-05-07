import shlex
import subprocess


def get_vm_state(vm_name: str) -> str:
    # showvminfo --machinereadable prints lines like: VMState="running"
    out = subprocess.check_output(
        ["VBoxManage", "showvminfo", vm_name, "--machinereadable"],
        text=True
    )
    for line in out.splitlines():
        if line.startswith("VMState="):
            return shlex.split(line.split("=",1)[1])[0]  # strip quotes
    raise RuntimeError(f"Could not determine state for VM {vm_name}")
