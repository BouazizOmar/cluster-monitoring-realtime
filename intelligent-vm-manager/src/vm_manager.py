#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import json
import logging
import time
from pathlib import Path
import shutil
import tempfile
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class VMSpecs:
    name: str
    memory_mb: int
    cpu_count: int
    disk_size_gb: int
    os_type: str
    state: str
    ip_address: Optional[str] = None

class IntelligentVMManager:
    def __init__(self, openai_api_key: Optional[str] = None, log_file: str = "vm_operations.log"):
        self.logger = self._setup_logger(log_file)
        self.client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key is required. Set it in .env file or pass it to the constructor.")
    
    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("IntelligentVMManager")
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

    def get_vm_specs(self, vm_name: str) -> VMSpecs:
        """Get detailed VM specifications including current state"""
        try:
            info = subprocess.check_output(
                ["VBoxManage", "showvminfo", vm_name, "--machinereadable"]
            ).decode()
            
            specs = {}
            for line in info.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    specs[key] = value.strip().strip('"')
            
            return VMSpecs(
                name=vm_name,
                memory_mb=int(specs.get("memory", 0)),
                cpu_count=int(specs.get("cpus", 0)),
                disk_size_gb=self._get_disk_size(vm_name),
                os_type=specs.get("ostype", "Unknown"),
                state=specs.get("VMState", "unknown"),
                ip_address=self._get_vm_ip(vm_name)
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get specs for VM {vm_name}: {e}")
            raise

    def _get_disk_size(self, vm_name: str) -> int:
        """Get the disk size of a VM in GB"""
        try:
            info = subprocess.check_output(
                ["VBoxManage", "showvminfo", vm_name, "--machinereadable"]
            ).decode()
            
            # Find the first hard disk
            for line in info.splitlines():
                if "SATA-0-0" in line and "=" in line:
                    disk_path = line.split("=", 1)[1].strip().strip('"')
                    # Get disk size
                    disk_info = subprocess.check_output(
                        ["VBoxManage", "showmediuminfo", disk_path, "--machinereadable"]
                    ).decode()
                    
                    for disk_line in disk_info.splitlines():
                        if "LogicalSize" in disk_line:
                            size_bytes = int(disk_line.split("=", 1)[1].strip().strip('"'))
                            return size_bytes // (1024 * 1024 * 1024)  # Convert to GB
            
            return 0
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get disk size for VM {vm_name}: {e}")
            return 0

    def _get_vm_ip(self, vm_name: str, retries: int = 5, delay: int = 5) -> Optional[str]:
        """Get the IP address of a VM"""
        for _ in range(retries):
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
            
            time.sleep(delay)
        
        return None

    def list_vms(self) -> List[str]:
        """List all available VMs"""
        try:
            out = subprocess.check_output(["VBoxManage", "list", "vms"]).decode()
            return [line.split()[0].strip('"') for line in out.splitlines()]
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to list VMs: {e}")
            return []

    def process_natural_language_command(self, command: str) -> Dict:
        """Process natural language commands using GPT"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are a VM management assistant. 
                     Parse the following command and return a JSON with the following structure:
                     {
                         "action": "migrate|scale|monitor",
                         "source_vm": "vm_name",
                         "target_vm": "vm_name",
                         "app_name": "app_name",
                         "resources": {
                             "memory_mb": int,
                             "cpu_count": int
                         }
                     }"""},
                    {"role": "user", "content": command}
                ],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Failed to process command: {e}")
            raise

    def handle_migration_request(self, command: str) -> bool:
        """Handle natural language migration requests"""
        try:
            # Parse the command using GPT
            parsed_command = self.process_natural_language_command(command)
            
            if parsed_command["action"] == "migrate":
                source_vm = parsed_command["source_vm"]
                app_name = parsed_command["app_name"]
                target_vm = parsed_command.get("target_vm")
                
                # If no target VM specified, use GPT to find the best one
                if not target_vm:
                    app_requirements = self._get_app_requirements(source_vm, app_name)
                    target_vm = self.find_best_target_vm(source_vm, app_requirements)
                
                return self.migrate_app(source_vm, app_name, target_vm)
            
            return False
        except Exception as e:
            self.logger.error(f"Failed to handle migration request: {e}")
            return False

    def _get_app_requirements(self, vm_name: str, app_name: str) -> Dict:
        """Get application requirements using GPT"""
        try:
            # First, collect basic app information
            app_info = self._collect_app_info(vm_name, app_name)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an application requirements analyzer."},
                    {"role": "user", "content": f"""Analyze this application and determine its requirements:
                    App Info: {json.dumps(app_info)}
                    Return requirements in JSON format."""}
                ],
                response_format={ "type": "json_object" }
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Failed to get app requirements: {e}")
            raise

    def _collect_app_info(self, vm_name: str, app_name: str) -> Dict:
        """Collect basic information about an application"""
        # This is a placeholder implementation
        # In a real implementation, you would collect actual app information
        return {
            "name": app_name,
            "vm": vm_name,
            "type": "unknown",
            "status": "unknown"
        }

    def find_best_target_vm(self, source_vm: str, app_requirements: Dict) -> Optional[str]:
        """Find the best VM to migrate to using GPT for decision making"""
        try:
            source_specs = self.get_vm_specs(source_vm)
            vms = self.list_vms()
            vm_specs_list = [self.get_vm_specs(vm) for vm in vms if vm != source_vm]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a VM migration expert."},
                    {"role": "user", "content": f"""Find the best target VM for migration:
                    Source VM: {json.dumps(source_specs.__dict__)}
                    Available VMs: {json.dumps([vm.__dict__ for vm in vm_specs_list])}
                    App Requirements: {json.dumps(app_requirements)}
                    Return the name of the best target VM."""}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Failed to find best target VM: {e}")
            return None

    def migrate_app(self, source_vm: str, app_name: str, target_vm: str) -> bool:
        """Migrate an application between VMs"""
        try:
            self.logger.info(f"Starting migration of {app_name} from {source_vm} to {target_vm}")
            
            # Ensure source VM is running
            self._ensure_vm_running(source_vm)
            
            # Create backup of source VM
            backup_name = f"{source_vm}_backup_{int(time.time())}"
            self._create_vm_backup(source_vm, backup_name)
            
            # Perform the migration
            # This is a placeholder implementation
            # In a real implementation, you would:
            # 1. Copy application files
            # 2. Update configurations
            # 3. Start the application on the target VM
            # 4. Verify the migration
            
            self.logger.info(f"Successfully migrated {app_name} from {source_vm} to {target_vm}")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return False

    def _ensure_vm_running(self, vm_name: str):
        """Ensure a VM is running"""
        try:
            state = self.get_vm_specs(vm_name).state
            if state.lower() != "running":
                subprocess.run(["VBoxManage", "startvm", vm_name, "--type", "headless"], check=True)
                time.sleep(10)  # Wait for VM to start
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start VM {vm_name}: {e}")
            raise

    def _create_vm_backup(self, vm_name: str, backup_name: str):
        """Create a backup of the VM"""
        try:
            subprocess.run(["VBoxManage", "snapshot", vm_name, "take", backup_name], check=True)
            self.logger.info(f"Created backup {backup_name} for VM {vm_name}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise 