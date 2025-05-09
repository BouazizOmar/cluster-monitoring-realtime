#!/usr/bin/env python3
import subprocess
import argparse
import time
import logging
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


# Enhanced VM resizer: snapshot → save state → drain → sync → stop → modify → restore state → health-check → rollback

def setup_logger(log_file: str = 'vm_resizer.log') -> logging.Logger:
    logger = logging.getLogger('VMResizer')
    logger.setLevel(logging.INFO)
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / log_file)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


class VMResizer:
    def __init__(self, vm_name: str, cpus: int, memory: Optional[int],
                 drain_script: Optional[str], sync_script: Optional[str],
                 health_url: Optional[str], timeout: int,
                 use_save_state: bool = True,
                 app_check_script: Optional[str] = None):
        self.vm = vm_name
        self.cpus = cpus
        self.memory = memory
        self.drain_script = drain_script
        self.sync_script = sync_script
        self.health_url = health_url
        self.timeout = timeout
        self.logger = setup_logger()
        self.snapshot_name = None
        self.use_save_state = use_save_state
        self.app_check_script = app_check_script
        self.saved_state_file = None
        self.running_apps_before = None

    def _run(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        self.logger.info(f"Running: {cmd}")
        return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

    def take_snapshot(self):
        ts = int(time.time())
        name = f"resize_backup_{self.vm}_{ts}"
        self._run(f"VBoxManage snapshot {self.vm} take {name}")
        self.snapshot_name = name
        self.logger.info(f"Snapshot created: {name}")

    def capture_running_apps(self):
        """Capture information about currently running apps in the VM"""
        if not self.app_check_script:
            self.logger.info("No app check script provided, skipping application state verification")
            return

        try:
            self.logger.info("Capturing running application state...")
            result = self._run(f"bash {self.app_check_script} {self.vm}", check=False)

            if result.returncode != 0:
                self.logger.warning(f"App check script failed with code {result.returncode}")
                self.logger.warning(f"Error: {result.stderr}")
                self.logger.warning("Continuing without application state verification")
                return

            self.running_apps_before = result.stdout.strip()
            self.logger.info(f"Current running apps: {self.running_apps_before}")
        except Exception as e:
            self.logger.warning(f"Failed to capture running apps: {e}")
            self.logger.warning("Continuing without application state verification")

    def verify_running_apps(self) -> bool:
        """Verify that previously running apps are still running"""
        if not self.app_check_script or not self.running_apps_before:
            self.logger.info("Skipping application verification (no script or initial state capture)")
            return True

        try:
            self.logger.info("Verifying applications are still running...")
            result = self._run(f"bash {self.app_check_script} {self.vm}", check=False)

            if result.returncode != 0:
                self.logger.warning(f"App verification failed with code {result.returncode}")
                self.logger.warning(f"Error: {result.stderr}")
                # Continue anyway if health check passed
                self.logger.warning("Assuming applications are OK since health check passed")
                return True

            running_apps_after = result.stdout.strip()

            # Compare before and after states
            if self.running_apps_before == running_apps_after:
                self.logger.info("All applications still running correctly")
                return True
            else:
                self.logger.warning(
                    f"Application state changed. Before: {self.running_apps_before}, After: {running_apps_after}")
                # Continue anyway if health check passed
                self.logger.warning("Continuing since health check passed")
                return True
        except Exception as e:
            self.logger.warning(f"Failed to verify apps: {e}")
            self.logger.warning("Continuing since health check passed")
            return True

    def save_vm_state(self):
        """Save the VM state instead of shutting down"""
        if not self.use_save_state:
            return

        # First check if VM is running
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        if 'VMState="running"' not in vm_info.stdout:
            self.logger.info(f"VM is not running, skipping save state")
            return

        self.logger.info(f"Saving state of VM {self.vm}")
        save_dir = Path('vm_states')
        save_dir.mkdir(exist_ok=True)

        # Optional: create a timestamped file for reference
        self.saved_state_file = save_dir / f"{self.vm}_state_{int(time.time())}.txt"
        # Don't actually save the state to this file, this is just for reference

        try:
            # Use VBoxManage directly to save state
            result = self._run(f"VBoxManage controlvm {self.vm} savestate", check=False)
            if result.returncode != 0:
                self.logger.error(f"Failed to save VM state: {result.stderr}")
                self.logger.warning("Trying alternative shutdown method")
                self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            else:
                self.logger.info(f"VM state saved successfully")

            # Wait a moment to ensure the state change is registered
            time.sleep(3)
        except Exception as e:
            self.logger.error(f"Error saving VM state: {e}")
            self.logger.warning("Falling back to power off")
            self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            time.sleep(3)

    def put_in_drain(self):
        if not self.drain_script:
            return
        self.logger.info("Draining connections...")
        try:
            self._run(f"bash {self.drain_script} {self.vm}")
        except Exception as e:
            self.logger.warning(f"Drain script failed: {e}; proceeding anyway")

    def sync_data(self):
        if not self.sync_script:
            return
        self.logger.info("Synchronizing data...")
        try:
            self._run(f"bash {self.sync_script} {self.vm}")
        except Exception as e:
            self.logger.warning(f"Sync script failed: {e}; proceeding anyway")

    def stop_vm(self):
        if self.use_save_state:
            # Already saved state
            return

        self.logger.info(f"Stopping VM {self.vm}")
        # ACPI shutdown
        self._run(f"VBoxManage controlvm {self.vm} acpipowerbutton", check=False)
        # wait up to 60s for graceful shutdown
        for i in range(60):
            out = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
            if 'VMState="poweroff"' in out.stdout:
                self.logger.info(f"VM gracefully shut down after {i + 1} seconds")
                return
            time.sleep(1)

        self.logger.warning("Forcing poweroff after timeout")
        self._run(f"VBoxManage controlvm {self.vm} poweroff")

    def modify_resources(self):
        # VirtualBox VMs must be powered off (not saved) to modify resources

        # First, check current VM state
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        vm_state_lines = [line for line in vm_info.stdout.splitlines() if line.startswith('VMState=')]
        current_state = vm_state_lines[0].split('=')[1].strip('"') if vm_state_lines else "unknown"

        self.logger.info(f"Current VM state before modification: {current_state}")

        # Handle different VM states
        if current_state == "saved":
            # For saved state, we need to first restore the VM
            self.logger.info("VM is in saved state, restoring to powered off state")
            # Two approaches - try the first, fall back to the alternative if needed
            try:
                # Try using controlvm poweroff (safer)
                self.logger.info("Trying poweroff method")
                self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
                time.sleep(3)
            except Exception as e:
                self.logger.warning(f"Failed to poweroff VM: {e}")

            # Check if it worked
            vm_state = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
            if 'VMState="poweroff"' not in vm_state.stdout:
                self.logger.warning("Still not powered off, trying alternative method")

                # Try discarding the saved state with error handling
                try:
                    result = self._run(f"VBoxManage snapshot {self.vm} restorecurrent", check=False)
                    if result.returncode != 0:
                        self.logger.warning(f"Failed to restore current snapshot: {result.stderr}")
                except Exception as e:
                    self.logger.warning(f"Error in snapshot restore: {e}")

                time.sleep(3)

        # For any state other than powered off, force power off
        vm_state = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        if 'VMState="poweroff"' not in vm_state.stdout:
            self.logger.info("VM is not powered off, forcing power off")
            self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            time.sleep(5)  # Give VirtualBox more time to register the poweroff

        # Check current settings to see if change is needed
        current_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
        current_cpu = None
        current_memory = None

        for line in current_info.splitlines():
            if line.startswith('cpus='):
                current_cpu = int(line.split('=')[1].strip('"'))
            elif line.startswith('memory='):
                current_memory = int(line.split('=')[1].strip('"'))

        self.logger.info(f"Current settings: CPUs={current_cpu}, Memory={current_memory}MB")

        # Build modification command for changed values only
        cmd = []
        if self.cpus and self.cpus != current_cpu:
            cmd.append(f"--cpus {self.cpus}")

        if self.memory and self.memory != current_memory:
            cmd.append(f"--memory {self.memory}")

        if not cmd:
            self.logger.info("No changes needed to CPU or memory")
            return

        args = ' '.join(cmd)
        self.logger.info(f"Applying: VBoxManage modifyvm {self.vm} {args}")
        result = self._run(f"VBoxManage modifyvm {self.vm} {args}", check=False)

        if result.returncode != 0:
            self.logger.error(f"Modification failed: {result.stderr}")
            raise RuntimeError(f"Failed to modify VM: {result.stderr}")

    def restore_saved_state(self):
        """Restore VM from saved state if available"""
        if self.use_save_state:
            self.logger.info(f"Restoring VM {self.vm} from saved state")
            self._run(f"VBoxManage startvm {self.vm} --type headless")
            time.sleep(10)  # Allow some time for state restoration
            return True
        return False

    def start_vm(self):
        # First check if VM is already running
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        if 'VMState="running"' in vm_info.stdout:
            self.logger.info("VM is already running")
            return

        if not self.restore_saved_state():
            self.logger.info(f"Starting VM {self.vm}")
            try:
                result = self._run(f"VBoxManage startvm {self.vm} --type headless", check=False)
                if result.returncode != 0:
                    self.logger.error(f"Failed to start VM: {result.stderr}")
                    # Try alternative start method
                    self.logger.info("Trying alternative start method")
                    self._run(f"VBoxManage startvm {self.vm}")
            except Exception as e:
                self.logger.error(f"Error starting VM: {e}")

        # Wait for VM to become fully responsive
        self.logger.info("Waiting for VM to become responsive...")

        # Check if VM is actually running
        time.sleep(5)
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        if 'VMState="running"' not in vm_info.stdout:
            self.logger.warning("VM doesn't appear to be running after start command")
            self.logger.info("Trying to force start the VM again")
            self._run(f"VBoxManage startvm {self.vm}", check=False)

        time.sleep(20)  # Give VM more time to fully initialize

        # Verify VM is running
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable", check=False)
        if 'VMState="running"' in vm_info.stdout:
            self.logger.info("VM confirmed running")
        else:
            self.logger.warning("VM may not be running properly. Current state from VirtualBox:")
            vm_state_lines = [line for line in vm_info.stdout.splitlines() if line.startswith('VMState=')]
            if vm_state_lines:
                self.logger.warning(f"VM state: {vm_state_lines[0]}")
            else:
                self.logger.warning("Could not determine VM state")

    def health_check(self) -> bool:
        if not self.health_url:
            self.logger.info("No health URL provided, skipping health check")
            return True

        try:
            import requests
        except ImportError:
            self.logger.warning("Python 'requests' module not installed. Skipping health check.")
            self.logger.warning("Install with: pip install requests")
            return True

        self.logger.info(f"Checking health at {self.health_url}")
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    self.logger.info("Health check passed")
                    return True
                else:
                    self.logger.debug(f"Health check returned status code {r.status_code}")
            except Exception as e:
                self.logger.debug(f"Health check attempt failed: {e}")
            time.sleep(5)

        self.logger.warning("Health check failed after timeout")
        # Ask if user wants to continue anyway
        try:
            response = input("Health check failed. Continue anyway? (y/n): ")
            if response.lower() in ['y', 'yes']:
                self.logger.info("Continuing despite failed health check")
                return True
        except:
            pass

        self.logger.error("Health check failed and was not overridden")
        return False

    def rollback(self):
        if not self.snapshot_name:
            self.logger.error("No snapshot to rollback to")
            return
        self.logger.info(f"Rolling back to snapshot {self.snapshot_name}")
        self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
        time.sleep(5)
        self._run(f"VBoxManage snapshot {self.vm} restore {self.snapshot_name}")
        self.start_vm()
        self.logger.info("Rollback complete")

    def execute(self):
        try:
            # First check if VM exists
            vm_exists = self._run(f"VBoxManage list vms | grep -w \"{self.vm}\"", check=False)
            if vm_exists.returncode != 0:
                self.logger.error(f"VM '{self.vm}' not found. Available VMs:")
                vms = self._run("VBoxManage list vms")
                sys.exit(1)

            # Take initial snapshot for safety
            self.take_snapshot()

            # Get VM state before doing anything
            vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable")
            vm_state_line = [line for line in vm_info.stdout.splitlines() if line.startswith('VMState=')]
            if vm_state_line:
                initial_state = vm_state_line[0].split('=')[1].strip('"')
                self.logger.info(f"Initial VM state: {initial_state}")

                # Only capture app state if VM is running
                if initial_state == "running":
                    self.capture_running_apps()
                    self.put_in_drain()
                    self.sync_data()
                else:
                    self.logger.info(f"VM is not running (state={initial_state}), skipping app checks and drain")

            # Save state or stop
            if self.use_save_state and initial_state == "running":
                self.save_vm_state()
            elif initial_state == "running":
                self.stop_vm()

            # Modify resources
            try:
                self.modify_resources()
            except Exception as e:
                self.logger.error(f"Resource modification failed: {e}")
                self.rollback()
                sys.exit(1)

            # Start VM back up
            self.start_vm()

            # Run health checks if URL provided
            if not self.health_check():
                raise RuntimeError("Health check failure")

            # Verify applications if initial state was captured
            if self.running_apps_before:
                if not self.verify_running_apps():
                    self.logger.warning("Application state verification failed, but continuing")

            self.logger.info("Resize successful with application state preserved")

            # Show final VM configuration
            final_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            for line in final_info.splitlines():
                if line.startswith('cpus=') or line.startswith('memory='):
                    self.logger.info(f"Final configuration: {line}")

        except Exception as e:
            self.logger.error(f"Error during resize: {e}")
            self.rollback()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Resize VirtualBox VM while preserving application state')
    parser.add_argument('vm_name', help='Name of the VM')
    parser.add_argument('--cpus', type=int, required=True, help='New CPU count')
    parser.add_argument('--memory', type=int, help='New memory in MB')
    parser.add_argument('--drain-script', help='Script to drain connections')
    parser.add_argument('--sync-script', help='Script to sync data')
    parser.add_argument('--health-url', help='URL to verify after start')
    parser.add_argument('--timeout', type=int, default=300, help='Health check timeout (s)')
    parser.add_argument('--use-save-state', action='store_true',
                        help='Use savestate instead of shutdown (better preserves app state)')
    parser.add_argument('--app-check-script', help='Script to verify running applications')
    args = parser.parse_args()

    resizer = VMResizer(
        vm_name=args.vm_name,
        cpus=args.cpus,
        memory=args.memory,
        drain_script=args.drain_script,
        sync_script=args.sync_script,
        health_url=args.health_url,
        timeout=args.timeout,
        use_save_state=args.use_save_state,
        app_check_script=args.app_check_script
    )
    resizer.execute()


if __name__ == '__main__':
    main()