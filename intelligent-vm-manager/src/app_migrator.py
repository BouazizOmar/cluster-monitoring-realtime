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
from typing import Optional, List, Dict
import openai
from dataclasses import dataclass
from enum import Enum

class AppType(Enum):
    FLASK = "flask"
    NODE = "node"
    SPRING = "spring"
    DJANGO = "django"
    REACT = "react"
    ANGULAR = "angular"
    VUE = "vue"
    UNKNOWN = "unknown"

@dataclass
class AppConfig:
    app_type: AppType
    exclude_patterns: List[str]
    pre_migration_commands: List[str]
    post_migration_commands: List[str]
    health_check_url: str
    health_check_port: int
    dependencies: List[str]

class AppMigrator:
    def __init__(self, log_file: str = "app_migration.log", openai_api_key: Optional[str] = None):
        self.logger = self._setup_logger(log_file)
        if openai_api_key:
            openai.api_key = openai_api_key
        self.app_configs = self._load_app_configs()
        
    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("AppMigrator")
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

    def _load_app_configs(self) -> Dict[str, AppConfig]:
        """Load predefined configurations for different application types"""
        return {
            AppType.FLASK.value: AppConfig(
                app_type=AppType.FLASK,
                exclude_patterns=["venv", "__pycache__", "*.pyc"],
                pre_migration_commands=["python3 -m venv venv", "source venv/bin/activate && pip install -r requirements.txt"],
                post_migration_commands=["source venv/bin/activate && python app.py"],
                health_check_url="/health",
                health_check_port=5040,
                dependencies=["python3", "pip"]
            ),
            AppType.NODE.value: AppConfig(
                app_type=AppType.NODE,
                exclude_patterns=["node_modules", "dist", "build"],
                pre_migration_commands=["npm install"],
                post_migration_commands=["npm start"],
                health_check_url="/api/health",
                health_check_port=3000,
                dependencies=["node", "npm"]
            ),
            # Add more predefined configs for other app types
        }

    def _detect_app_type(self, app_path: str) -> AppType:
        """Use AI to detect the application type based on files and structure"""
        try:
            # Get list of files in the application directory
            files = self._run_command(f"ssh -p 2222 omar@127.0.0.1 'find {app_path} -type f -not -path \"*/\.*\"'")
            
            # Use OpenAI to analyze the files and determine app type
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at detecting application types based on file structure."},
                    {"role": "user", "content": f"Based on these files, what type of application is this? Files:\n{files}"}
                ]
            )
            
            detected_type = response.choices[0].message.content.lower()
            
            # Map the detected type to our AppType enum
            for app_type in AppType:
                if app_type.value in detected_type:
                    return app_type
            
            return AppType.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Failed to detect app type: {e}")
            return AppType.UNKNOWN

    def _analyze_dependencies(self, app_path: str) -> List[str]:
        """Use AI to analyze and suggest required dependencies"""
        try:
            # Get content of key files
            files_content = {}
            for file in ["requirements.txt", "package.json", "pom.xml", "build.gradle"]:
                try:
                    content = self._run_command(f"ssh -p 2222 omar@127.0.0.1 'cat {app_path}/{file}'")
                    files_content[file] = content
                except:
                    continue
            
            # Use OpenAI to analyze dependencies
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing application dependencies."},
                    {"role": "user", "content": f"Based on these files, what system dependencies are required? Files:\n{json.dumps(files_content)}"}
                ]
            )
            
            return response.choices[0].message.content.split("\n")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze dependencies: {e}")
            return []

    def _generate_migration_plan(self, app_path: str, detected_type: AppType) -> AppConfig:
        """Use AI to generate a custom migration plan"""
        try:
            # Get application structure and key files
            structure = self._run_command(f"ssh -p 2222 omar@127.0.0.1 'find {app_path} -type f -not -path \"*/\.*\"'")
            
            # Use OpenAI to generate migration plan
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating application migration plans."},
                    {"role": "user", "content": f"Create a migration plan for this {detected_type.value} application. Structure:\n{structure}"}
                ]
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            return AppConfig(
                app_type=detected_type,
                exclude_patterns=plan.get("exclude_patterns", []),
                pre_migration_commands=plan.get("pre_migration_commands", []),
                post_migration_commands=plan.get("post_migration_commands", []),
                health_check_url=plan.get("health_check_url", "/health"),
                health_check_port=plan.get("health_check_port", 8080),
                dependencies=plan.get("dependencies", [])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate migration plan: {e}")
            return self.app_configs.get(detected_type.value, self.app_configs[AppType.UNKNOWN.value])

    def migrate_app(self, 
                   source_vm: str, 
                   target_vm: str, 
                   app_path: str = "/opt/sampleapp",
                   app_type: Optional[str] = None,
                   exclude_patterns: Optional[List[str]] = None,
                   pre_migration_commands: Optional[List[str]] = None,
                   post_migration_commands: Optional[List[str]] = None,
                   health_check_url: Optional[str] = None,
                   health_check_port: Optional[int] = None):
        """Migrate an application between VMs with AI assistance"""
        try:
            self.logger.info(f"Starting migration from {source_vm} to {target_vm}")
            
            # 1. Ensure both VMs are running
            self._ensure_vm_running(source_vm)
            self._ensure_vm_running(target_vm)
            
            # 2. Set up port forwarding
            self._setup_port_forwarding(source_vm, 2222, 22)
            self._setup_port_forwarding(target_vm, 2223, 22)
            
            # 3. Create backup
            backup_name = self._create_backup(source_vm)
            
            # 4. Detect application type if not specified
            detected_type = AppType(app_type) if app_type else self._detect_app_type(app_path)
            self.logger.info(f"Detected application type: {detected_type.value}")
            
            # 5. Generate or use provided migration configuration
            if all([exclude_patterns, pre_migration_commands, post_migration_commands, 
                   health_check_url, health_check_port]):
                config = AppConfig(
                    app_type=detected_type,
                    exclude_patterns=exclude_patterns,
                    pre_migration_commands=pre_migration_commands,
                    post_migration_commands=post_migration_commands,
                    health_check_url=health_check_url,
                    health_check_port=health_check_port,
                    dependencies=[]
                )
            else:
                config = self._generate_migration_plan(app_path, detected_type)
            
            # 6. Check and install dependencies
            dependencies = self._analyze_dependencies(app_path)
            for dep in dependencies:
                self._run_command(f"ssh -p 2223 omar@127.0.0.1 'which {dep} || sudo apt-get install -y {dep}'")
            
            # 7. Run pre-migration commands
            if config.pre_migration_commands:
                self.logger.info("Running pre-migration commands")
                for cmd in config.pre_migration_commands:
                    self._run_command(f"ssh -p 2222 omar@127.0.0.1 '{cmd}'")
            
            # 8. Copy application files
            with tempfile.TemporaryDirectory() as tmp_dir:
                exclude_args = " ".join([f"--exclude='{pattern}'" for pattern in config.exclude_patterns])
                
                self._run_command(
                    f"rsync -avz -e 'ssh -p 2222' {exclude_args} "
                    f"omar@127.0.0.1:{app_path}/ {tmp_dir}/"
                )
                
                self._run_command(
                    f"rsync -avz -e 'ssh -p 2223' {tmp_dir}/ "
                    f"omar@127.0.0.1:{app_path}/"
                )
            
            # 9. Run post-migration commands
            if config.post_migration_commands:
                self.logger.info("Running post-migration commands")
                for cmd in config.post_migration_commands:
                    self._run_command(f"ssh -p 2223 omar@127.0.0.1 '{cmd}'")
            
            # 10. Verify application health
            if config.health_check_url and config.health_check_port:
                time.sleep(5)
                self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 'curl -s http://localhost:{config.health_check_port}{config.health_check_url}'"
                )
            
            self.logger.info("Migration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            if 'backup_name' in locals():
                self.logger.info("Attempting rollback...")
                try:
                    subprocess.run(["VBoxManage", "controlvm", source_vm, "poweroff"], check=True)
                    time.sleep(5)
                    subprocess.run(["VBoxManage", "snapshot", source_vm, "restore", backup_name], check=True)
                    self.logger.info("Rollback successful")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Rollback failed: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Application Migrator")
    parser.add_argument("source_vm", help="Source VM name")
    parser.add_argument("target_vm", help="Target VM name")
    parser.add_argument("--app-path", default="/opt/sampleapp", 
                       help="Application path (default: /opt/sampleapp)")
    parser.add_argument("--app-type", 
                       help="Type of application (flask, node, spring, etc.)")
    parser.add_argument("--exclude", nargs="+",
                       help="Patterns to exclude from migration")
    parser.add_argument("--pre-migration", nargs="+",
                       help="Commands to run before migration")
    parser.add_argument("--post-migration", nargs="+",
                       help="Commands to run after migration")
    parser.add_argument("--health-check-url",
                       help="URL to check application health")
    parser.add_argument("--health-check-port", type=int,
                       help="Port to check application health")
    parser.add_argument("--openai-api-key",
                       help="OpenAI API key for AI features")
    
    args = parser.parse_args()
    
    migrator = AppMigrator(openai_api_key=args.openai_api_key)
    success = migrator.migrate_app(
        args.source_vm,
        args.target_vm,
        args.app_path,
        args.app_type,
        args.exclude,
        args.pre_migration,
        args.post_migration,
        args.health_check_url,
        args.health_check_port
    )
    
    if success:
        print("Migration completed successfully")
    else:
        print("Migration failed")
        exit(1)

if __name__ == "__main__":
    main() 