#!/usr/bin/env python3
import subprocess
import argparse
import time
import logging
import sys
import os
from pathlib import Path
import requests
from typing import Optional, Dict, Any

# VM states
VM_STATE_POWEROFF = "poweroff"
VM_STATE_RUNNING = "running"
VM_STATE_SAVED = "saved"
VM_STATE_PAUSED = "paused"

# Default SSH settings
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_TIMEOUT = 30

def setup_logger(log_file: str = 'vm_resizer.log') -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    logger = logging.getLogger('VMResizer')
    logger.setLevel(logging.DEBUG)

    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # File handler
    fh = logging.FileHandler(log_dir / log_file)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

class VMResizer:
    def __init__(
            self,
            vm_name: str,
            cpus: int,
            memory: int,
            health_url: Optional[str] = None,
            timeout: int = 300,
            debug: bool = False,
            wait_timeout: int = 60,
            ssh_config: Optional[Dict[str, Any]] = None,
            app_config: Optional[Dict[str, Any]] = None
    ):
        self.vm = vm_name
        self.cpus = cpus
        self.memory = memory
        self.health_url = health_url
        self.timeout = timeout
        self.logger = setup_logger()
        self.debug = debug
        self.wait_timeout = wait_timeout
        
        # Application configuration
        self.app_config = app_config or {}
        self.app_name = self.app_config.get('name', '')
        self.app_start_command = self.app_config.get('start_command', '')
        self.app_status_command = self.app_config.get('status_command', '')
        self.app_process_name = self.app_config.get('process_name', '')

    def _run(self, cmd: str, check: bool = True, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a command with proper error handling."""
        self.logger.info(f"Running: {cmd}")
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.stdout and self.debug:
                self.logger.debug(f"Command stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Command stderr: {result.stderr}")
                
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=cmd,
                    output=result.stdout,
                    stderr=result.stderr
                )
                
            return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out after {timeout} seconds: {cmd}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing command '{cmd}': {str(e)}")
            raise

    def get_vm_state(self) -> str:
        """Get the current state of the VM."""
        try:
            info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            for line in info.splitlines():
                if line.startswith('VMState='):
                    return line.split('=')[1].strip('"')
            return "unknown"
        except Exception as e:
            self.logger.error(f"Error getting VM state: {str(e)}")
            return "error"

    def wait_for_state(self, target_state: str, timeout: int = 60) -> bool:
        """Wait for VM to reach a specific state within timeout."""
        self.logger.info(f"Waiting for VM to reach '{target_state}' state")
        end_time = time.time() + timeout

        while time.time() < end_time:
            current_state = self.get_vm_state()
            if current_state == target_state:
                return True
            time.sleep(2)

        self.logger.error(f"Timeout waiting for VM to reach '{target_state}' state")
        return False

    def power_off_vm(self) -> bool:
        """Power off the VM."""
        state = self.get_vm_state()
        if state == VM_STATE_POWEROFF:
            self.logger.info("VM is already powered off")
            return True

        try:
            if state == VM_STATE_SAVED:
                self._run(f"VBoxManage discardstate {self.vm}", check=False)
            elif state == VM_STATE_PAUSED:
                self._run(f"VBoxManage controlvm {self.vm} resume", check=False)
                time.sleep(2)
            
            self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            return self.wait_for_state(VM_STATE_POWEROFF, timeout=self.wait_timeout)
        except Exception as e:
            self.logger.error(f"Error powering off VM: {e}")
            return False

    def get_current_cpus(self) -> int:
        """Get the current CPU count for the VM."""
        try:
            info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            for line in info.splitlines():
                if line.startswith('cpus='):
                    return int(line.split('=')[1].strip('"'))
            return 0
        except Exception as e:
            self.logger.error(f"Error getting CPU count: {str(e)}")
            return 0

    def get_current_memory(self) -> int:
        """Get the current memory allocation for the VM."""
        try:
            info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            for line in info.splitlines():
                if line.startswith('memory='):
                    return int(line.split('=')[1].strip('"'))
            return 0
        except Exception as e:
            self.logger.error(f"Error getting memory allocation: {str(e)}")
            return 0

    def modify_cpu(self) -> bool:
        """Modify the CPU count for the VM."""
        if self.get_vm_state() != VM_STATE_POWEROFF:
            self.logger.error("Cannot modify CPU when VM is not powered off")
            return False

        current = self.get_current_cpus()
        if self.cpus != current:
            self.logger.info(f"Changing CPU from {current} to {self.cpus} cores")
            try:
                self._run(f"VBoxManage modifyvm {self.vm} --cpus {self.cpus}")
                return self.get_current_cpus() == self.cpus
            except Exception as e:
                self.logger.error(f"Error modifying CPU: {e}")
                return False
        return True

    def modify_memory(self) -> bool:
        """Modify the memory allocation for the VM."""
        if self.get_vm_state() != VM_STATE_POWEROFF:
            self.logger.error("Cannot modify memory when VM is not powered off")
            return False

        current = self.get_current_memory()
        if self.memory != current:
            self.logger.info(f"Changing memory from {current} to {self.memory} MB")
            try:
                self._run(f"VBoxManage modifyvm {self.vm} --memory {self.memory}")
                return self.get_current_memory() == self.memory
            except Exception as e:
                self.logger.error(f"Error modifying memory: {e}")
                return False
        return True

    def start_vm(self, start_type: str = "headless") -> bool:
        """Start the VM."""
        self.logger.info(f"Starting VM {self.vm} in {start_type} mode")
        try:
            self._run(f"VBoxManage startvm {self.vm} --type {start_type}")
            return self.wait_for_state(VM_STATE_RUNNING, timeout=self.wait_timeout)
        except Exception as e:
            self.logger.error(f"Error starting VM: {e}")
            return False

    def health_check(self) -> bool:
        """Perform basic health check on the VM."""
        if self.get_vm_state() != VM_STATE_RUNNING:
            self.logger.error("Health check failed: VM is not running")
            return False

        if self.health_url:
            self.logger.info(f"Checking health at {self.health_url}")
            deadline = time.time() + self.timeout
            
            while time.time() < deadline:
                try:
                    r = requests.get(self.health_url, timeout=5)
                    if r.status_code == 200:
                        self.logger.info("Health check passed")
                        return True
                except Exception as e:
                    self.logger.debug(f"Health check request failed: {e}")
                time.sleep(5)
            
            self.logger.error("Health check failed after timeout")
            return False
            
        return True

    def execute(self):
        """Main execution flow for VM resizing operation."""
        try:
            # Check if VM needs to be modified
            current_cpus = self.get_current_cpus()
            current_mem = self.get_current_memory()
            self.logger.info(f"Current VM configuration: CPUs={current_cpus}, Memory={current_mem} MB")

            if current_cpus == self.cpus and current_mem == self.memory:
                self.logger.info(f"VM already has {self.cpus} CPUs and {self.memory} MB memory. No changes needed.")
                return

            # Power off VM to make changes
            if not self.power_off_vm():
                self.logger.error("Failed to power off VM")
                sys.exit(1)

            # Make hardware changes
            cpu_changed = self.modify_cpu()
            mem_changed = self.modify_memory()

            if not (cpu_changed and mem_changed):
                self.logger.error("Failed to modify VM hardware")
                sys.exit(1)

            # Start the VM
            if not self.start_vm():
                self.logger.error("Failed to start VM after resize")
                sys.exit(1)

            # Wait for VM to stabilize
            time.sleep(30)

            # Perform health check
            if not self.health_check():
                self.logger.error("Health check failed after resize")
                sys.exit(1)

            self.logger.info("Resize operation completed successfully")

        except Exception as e:
            self.logger.error(f"Resize error: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Resize VirtualBox VM')
    parser.add_argument('vm_name', help='Name of the VM')
    parser.add_argument('--cpus', type=int, required=True, help='New CPU count')
    parser.add_argument('--memory', type=int, required=True, help='New memory in MB')
    parser.add_argument('--health-url', help='URL to verify application state (optional)')
    parser.add_argument('--timeout', type=int, default=300, help='Health check timeout (s)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--wait-timeout', type=int, default=60, help='Timeout for state transitions (s)')
    
    # Application management parameters
    parser.add_argument('--app-name', help='Application name for logging')
    parser.add_argument('--app-process', help='Process name to check if application is running')
    parser.add_argument('--app-start-cmd', help='Command to start the application')
    parser.add_argument('--app-status-cmd', help='Command to check application status')

    args = parser.parse_args()
    
    # Build application configuration if provided
    app_config = None
    if args.app_name or args.app_process or args.app_start_cmd:
        app_config = {
            'name': args.app_name or 'application',
            'process_name': args.app_process,
            'start_command': args.app_start_cmd,
            'status_command': args.app_status_cmd
        }

    resizer = VMResizer(
        vm_name=args.vm_name,
        cpus=args.cpus,
        memory=args.memory,
        health_url=args.health_url,
        timeout=args.timeout,
        debug=args.debug,
        wait_timeout=args.wait_timeout,
        app_config=app_config
    )

    resizer.execute()

if __name__ == '__main__':
    main()