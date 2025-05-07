import subprocess
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

INVENTORY = os.getenv("ANSIBLE_INVENTORY", "inventory.ini")
PLAYBOOK  = os.getenv("ANSIBLE_PLAYBOOK", "migrate_app.yml")

def run(cmd: list[str]):
    logging.info(f"RUN: {' '.join(cmd)}")
    if cmd[0] == "ansible-playbook":
        cmd = ["ansible-playbook", "-c", "local", "-vvv"] + cmd[1:]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        message = "\n".join(filter(None, [out, err]))
        logging.error(message)
        raise RuntimeError(message)
    if res.stderr:
        logging.warning(res.stderr.strip())
    return res.stdout.strip()

class Handlers:
    @staticmethod
    def start_vm(plan):
        return run(["VBoxManage", "startvm", plan.vm, "--type", "headless"])

    @staticmethod
    def stop_vm(plan):
        return run(["VBoxManage", "controlvm", plan.vm, "poweroff"])

    @staticmethod
    def restart_vm(plan):
        Handlers.stop_vm(plan)
        return Handlers.start_vm(plan)

    @staticmethod
    def scale_vm(plan):
        mem  = plan.params["memory_mb"]
        cpus = plan.params["cpus"]
        return run([
            "ansible-playbook", "-i", INVENTORY, os.getenv("ANSIBLE_SCALE_PLAYBOOK", "scale_vm.yml"),
            "-e", f"vm_name={plan.vm}",
            "-e", f"new_ram_mb={mem}",
            "-e", f"new_cpu_count={cpus}"
        ])

    @staticmethod
    def migrate_app(plan):
        src = plan.params["src_vm"]
        dest = plan.params["dest_vm"]
        app_path = plan.params.get("app_path")
        service = plan.params.get("app_service")
        setup_cmd = plan.params.get("setup_command")
        start_cmd = plan.params.get("start_command")

        extra_vars = []
        for key, val in {
            'src': src,
            'dest': dest,
            'app_path': app_path,
            'app_service': service,
            'setup_command': setup_cmd,
            'start_command': start_cmd
        }.items():
            if val:
                extra_vars.extend(["-e", f"{key}={val}"])

        cmd = [
            "ansible-playbook", "-i", INVENTORY, PLAYBOOK,
            *extra_vars
        ]
        return run(cmd)
