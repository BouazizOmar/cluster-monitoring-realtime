#!/usr/bin/env python3
import subprocess
import shlex
import time
import os
import json
import logging
from pathlib import Path
import tempfile
import shutil

class FlaskMigrator:
    def __init__(self, log_file: str = "flask_migration.log"):
        self.logger = self._setup_logger(log_file)
        
    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("FlaskMigrator")
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_dir / log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _run_command(self, cmd: str, check: bool = True) -> str:
        """Run a command and return its output"""
        self.logger.info(f"Running command: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, check=check, 
                                 capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.stderr}")
            raise

    def _get_vm_ip(self, vm_name: str) -> str:
        """Get VM's IP address"""
        for _ in range(5):  # Try 5 times
            try:
                out = subprocess.check_output([
                    "VBoxManage", "guestproperty", "get", vm_name,
                    "/VirtualBox/GuestInfo/Net/0/V4/IP"
                ]).decode().strip()
                
                if "Value:" in out and "no value" not in out.lower():
                    _, ip = out.split("Value:", 1)
                    return ip.strip()
            except subprocess.CalledProcessError:
                pass
            
            time.sleep(5)
        
        raise RuntimeError(f"Could not get IP for VM {vm_name}")

    def _ensure_vm_running(self, vm_name: str):
        """Ensure VM is running"""
        try:
            state = subprocess.check_output([
                "VBoxManage", "showvminfo", vm_name, "--machinereadable"
            ]).decode()
            
            if "VMState=\"running\"" not in state:
                self.logger.info(f"Starting VM {vm_name}")
                subprocess.run(["VBoxManage", "startvm", vm_name, "--type", "headless"], 
                             check=True)
                time.sleep(10)  # Wait for VM to start
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start VM {vm_name}: {e}")
            raise

    def _create_backup(self, vm_name: str):
        """Create a backup of the VM"""
        backup_name = f"{vm_name}_backup_{int(time.time())}"
        try:
            subprocess.run(["VBoxManage", "snapshot", vm_name, "take", backup_name], 
                         check=True)
            self.logger.info(f"Created backup {backup_name} for VM {vm_name}")
            return backup_name
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise

    def _setup_port_forwarding(self, vm_name: str, host_port: int, guest_port: int):
        """Set up port forwarding for a VM"""
        try:
            # First, try to remove any existing migration_ssh rule
            try:
                self.logger.info(f"Removing existing migration_ssh rule for {vm_name}")
                subprocess.run([
                    "VBoxManage", "controlvm", vm_name,
                    "natpf1", "delete", "migration_ssh"
                ], check=False)  # Don't check for errors as the rule might not exist
                time.sleep(1)  # Wait for rule to be removed
            except subprocess.CalledProcessError:
                pass  # Ignore errors when removing non-existent rules
            
            # Now add the new port forwarding rule
            self.logger.info(f"Setting up port forwarding for {vm_name}")
            subprocess.run([
                "VBoxManage", "controlvm", vm_name,
                "natpf1", f"migration_ssh,tcp,127.0.0.1,{host_port},,{guest_port}"
            ], check=True)
            time.sleep(2)  # Wait for port forwarding to take effect
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set up port forwarding: {e}")
            raise

    def migrate_flask_app(self, source_vm: str, target_vm: str, app_path: str = "/opt/sampleapp"):
        """Migrate a Flask application between VMs"""
        try:
            self.logger.info(f"Starting migration of Flask app from {source_vm} to {target_vm}")
            
            # 1. Ensure both VMs are running
            self._ensure_vm_running(source_vm)
            self._ensure_vm_running(target_vm)
            
            # 2. Set up port forwarding using different ports
            self._setup_port_forwarding(source_vm, 2222, 22)  # Source VM SSH
            self._setup_port_forwarding(target_vm, 2223, 22)  # Target VM SSH
            
            # 3. Create backup of source VM
            backup_name = self._create_backup(source_vm)
            
            # 4. Create temporary directory for staging
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 5. Copy application files from source to staging (excluding venv)
                self.logger.info("Copying application files from source VM")
                try:
                    # Copy app.py
                    self._run_command(
                        f"scp -P 2222 omar@127.0.0.1:{app_path}/app.py {tmp_dir}/"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to copy app.py: {e}")
                    raise
                
                try:
                    # Copy requirements.txt
                    self._run_command(
                        f"scp -P 2222 omar@127.0.0.1:{app_path}/requirements.txt {tmp_dir}/"
                    )
                except Exception as e:
                    self.logger.warning(f"requirements.txt not found: {e}")
                    # Create requirements.txt with Flask dependency
                    with open(f"{tmp_dir}/requirements.txt", "w") as f:
                        f.write("Flask==3.1.0\n")
                
                # 6. Copy application files to target VM
                self.logger.info("Copying application files to target VM")
                self._run_command(
                    f"scp -r -P 2223 {tmp_dir}/* omar@127.0.0.1:{app_path}/"
                )
            
            # 7. Set up virtual environment and install dependencies on target VM
            self.logger.info("Setting up virtual environment and installing dependencies")
            self._run_command(
                f"ssh -p 2223 omar@127.0.0.1 'cd {app_path} && "
                "rm -rf venv && "  # Remove existing venv
                "python3 -m venv venv && "
                "source venv/bin/activate && "
                "pip install -r requirements.txt'"
            )
            
            # 8. Start the application on target VM
            self.logger.info("Starting application on target VM")
            self._run_command(
                f"ssh -p 2223 omar@127.0.0.1 'cd {app_path} && "
                "source venv/bin/activate && "
                "nohup python app.py > app.log 2>&1 &'"
            )
            
            # 9. Verify the application is running
            time.sleep(5)  # Wait for app to start
            self._run_command(
                f"ssh -p 2223 omar@127.0.0.1 'curl -s http://localhost:5040'"
            )
            
            self.logger.info("Migration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            # Attempt rollback if backup was created
            if 'backup_name' in locals():
                self.logger.info("Attempting rollback...")
                try:
                    # Power off the VM before restoring snapshot
                    subprocess.run(["VBoxManage", "controlvm", source_vm, "poweroff"], 
                                 check=True)
                    time.sleep(5)  # Wait for VM to power off
                    
                    # Restore the snapshot
                    subprocess.run(["VBoxManage", "snapshot", source_vm, "restore", backup_name], 
                                 check=True)
                    self.logger.info("Rollback successful")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Rollback failed: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Flask Application Migrator")
    parser.add_argument("source_vm", help="Source VM name")
    parser.add_argument("target_vm", help="Target VM name")
    parser.add_argument("--app-path", default="/opt/sampleapp", 
                       help="Application path (default: /opt/sampleapp)")
    
    args = parser.parse_args()
    
    migrator = FlaskMigrator()
    success = migrator.migrate_flask_app(args.source_vm, args.target_vm, args.app_path)
    
    if success:
        print("Migration completed successfully")
    else:
        print("Migration failed")
        exit(1)

if __name__ == "__main__":
    main() 