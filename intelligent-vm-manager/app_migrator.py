#!/usr/bin/env python3
import subprocess
import argparse
import time
import logging
import sys
from pathlib import Path
import requests
import openai
from dotenv import load_dotenv
import os
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from openai import OpenAI
import shlex
import re
import tempfile
import shutil

# Load environment variables from .env file
load_dotenv()

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
    def __init__(
        self,
        vm_name: str,
        cpus: int,
        memory: int,
        health_url: str = None,
        timeout: int = 300,
        openai_api_key: str = None
    ):
        self.vm = vm_name
        self.cpus = cpus
        self.memory = memory
        self.health_url = health_url
        self.timeout = timeout
        self.logger = setup_logger()
        self.snapshot_name = None
        # only configure openai if we actually have a key
        if openai_api_key:
            openai.api_key = openai_api_key
            self.client = openai
        else:
            self.client = None

    def _run(self, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        self.logger.info(f"Running: {cmd}")
        return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)

    def take_snapshot(self):
        ts = int(time.time())
        name = f"resize_backup_{self.vm}_{ts}"
        self._run(f"VBoxManage snapshot {self.vm} take {name}")
        self.snapshot_name = name
        self.logger.info(f"Snapshot created: {name}")

    def save_vm_state(self):
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable")
        if 'VMState="running"' not in vm_info.stdout:
            self.logger.info("VM is not running, skipping save state")
            return
        self.logger.info(f"Saving state of VM {self.vm}")
        self._run(f"VBoxManage controlvm {self.vm} savestate")
        time.sleep(5)  # Wait for state to be fully saved

    def power_off_if_needed(self):
        vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable")
        current_state = [
            line.split('=')[1].strip('"')
            for line in vm_info.stdout.splitlines()
            if line.startswith('VMState=')
        ][0]
        if current_state == "saved":
            self.logger.info("VM is in saved state, powering off to allow modification")
            self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            time.sleep(5)
        elif current_state != "poweroff":
            self.logger.info(f"VM is in {current_state} state, forcing power off")
            self._run(f"VBoxManage controlvm {self.vm} poweroff", check=False)
            time.sleep(5)

    def modify_cpu(self):
        current_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
        current_cpu = int([
            line.split('=')[1].strip('"')
            for line in current_info.splitlines()
            if line.startswith('cpus=')
        ][0])
        if self.cpus != current_cpu:
            self._run(f"VBoxManage modifyvm {self.vm} --cpus {self.cpus}")
            self.logger.info(f"CPU adjusted to {self.cpus} cores")

    def modify_memory(self):
        current_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
        current_memory = int([
            line.split('=')[1].strip('"')
            for line in current_info.splitlines()
            if line.startswith('memory=')
        ][0])
        if self.memory != current_memory:
            self._run(f"VBoxManage modifyvm {self.vm} --memory {self.memory}")
            self.logger.info(f"Memory adjusted to {self.memory} MB")

    def start_vm(self):
        self.logger.info(f"Starting VM {self.vm}")
        self._run(f"VBoxManage startvm {self.vm} --type headless")
        time.sleep(10)  # Allow time for VM to boot

    def health_check(self) -> bool:
        if not self.health_url:
            self.logger.info("No health URL provided, assuming success")
            return True
        self.logger.info(f"Checking health at {self.health_url}")
        start = time.time()
        while time.time() - start < self.timeout:
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    self.logger.info("Health check passed")
                    return True
                self.logger.debug(f"Health check returned {r.status_code}")
            except Exception as e:
                self.logger.debug(f"Health check attempt failed: {e}")
            time.sleep(5)
        self.logger.error("Health check failed after timeout")
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

    def analyze_with_openai(self, log_data: str):
        if not self.client:
            self.logger.info("OpenAI client not initialized, skipping analysis")
            return
        system = {
            "role": "system",
            "content": "Analyze the following log data from a VM scaling operation and suggest actions if issues are detected."
        }
        user_message = {
            "role": "user",
            "content": log_data
        }
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[system, user_message],
                max_tokens=150
            )
            analysis = response.choices[0].message.content.strip()
            self.logger.info(f"OpenAI Analysis: {analysis}")
        except Exception as e:
            self.logger.error(f"Failed to analyze with OpenAI: {e}")

    def execute(self):
        try:
            # Verify VM exists
            vm_list = self._run("VBoxManage list vms").stdout
            if self.vm not in vm_list:
                self.logger.error(f"VM '{self.vm}' not found")
                sys.exit(1)

            # Take snapshot for safety
            self.take_snapshot()

            # Check if VM is running for CPU hot-plugging
            vm_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            is_running = 'VMState="running"' in vm_info

            # Handle CPU scaling (hot-plugging if running)
            if is_running:
                self.modify_cpu()
            else:
                self.save_vm_state()
                self.power_off_if_needed()
                self.modify_cpu()
                self.modify_memory()
                self.start_vm()

            # Handle RAM scaling (requires state save/restore)
            if not is_running or self.memory != int([
                line.split('=')[1].strip('"')
                for line in vm_info.splitlines()
                if line.startswith('memory=')
            ][0]):
                self.save_vm_state()
                self.power_off_if_needed()
                self.modify_memory()
                self.start_vm()

            # Verify application state
            if not self.health_check():
                raise RuntimeError("Health check failed")

            self.logger.info("Resize successful")
            final_info = self._run(f"VBoxManage showvminfo {self.vm} --machinereadable").stdout
            for line in final_info.splitlines():
                if line.startswith('cpus=') or line.startswith('memory='):
                    self.logger.info(f"Final configuration: {line}")

            # Analyze logs with OpenAI
            with open('logs/vm_resizer.log', 'r') as f:
                log_data = f.read()
            self.analyze_with_openai(log_data)

        except Exception as e:
            self.logger.error(f"Error during resize: {e}")
            self.rollback()
            sys.exit(1)

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
    health_check_port: Optional[int]
    dependencies: List[str]
    app_entrypoint: Optional[str] = None

@dataclass
class MigrationState:
    source_vm: str
    target_vm: str
    application: str
    port: Optional[int]
    status: str
    timestamp: str
    error: Optional[str] = None
    app_type: Optional[AppType] = None
    app_port: Optional[int] = None

class AppMigrator:
    def __init__(
        self,
        source_vm: str,
        target_vm: str,
        application: str,
        port: Optional[int] = None,
        health_url: Optional[str] = None,
        timeout: int = 300,
        debug: bool = False,
        openai_api_key: Optional[str] = None
    ):
        self.source_vm = source_vm
        self.target_vm = target_vm
        self.application = application
        self.port = port
        self.health_url = health_url
        self.timeout = timeout
        self.logger = self._setup_logging(debug)
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.app_configs = self._load_app_configs()
    
    def _setup_logging(self, debug: bool) -> logging.Logger:
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app_migration.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_app_configs(self) -> Dict[str, AppConfig]:
        return {
            AppType.FLASK.value: AppConfig(
                app_type=AppType.FLASK,
                exclude_patterns=["venv", "__pycache__", "*.pyc", "*.pyo", "*.pyd"],
                pre_migration_commands=["python3 -m venv venv",
                                      "source venv/bin/activate && pip install -r requirements.txt"],
                post_migration_commands=[],
                health_check_url="/health",
                health_check_port=None,
                dependencies=["python3", "pip", "python3-venv"]
            ),
            AppType.NODE.value: AppConfig(
                app_type=AppType.NODE,
                exclude_patterns=["node_modules", "dist", "build"],
                pre_migration_commands=["npm install"],
                post_migration_commands=["npm start"],
                health_check_url="/api/health",
                health_check_port=None,
                dependencies=["node", "npm"]
            ),
            AppType.UNKNOWN.value: AppConfig(
                app_type=AppType.UNKNOWN,
                exclude_patterns=["node_modules", "venv", "__pycache__", "*.pyc", "dist", "build"],
                pre_migration_commands=[],
                post_migration_commands=[],
                health_check_url="/health",
                health_check_port=8080,
                dependencies=[]
            )
        }

    async def _run_command(self, cmd: str, check: bool = True) -> str:
        """Run a command and return its output"""
        self.logger.info(f"Running command: {cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if check and process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, stderr.decode())
            
            return stdout.decode()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.stderr}")
            raise

    async def verify_vm_running(self, vm_name: str) -> bool:
        """Verify that a VM is running"""
        try:
            cmd = ["VBoxManage", "showvminfo", vm_name, "--machinereadable"]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"Failed to get VM state: {stderr.decode()}")
                return False
            
            # Check if VM is running
            for line in stdout.decode().split('\n'):
                if line.startswith('VMState='):
                    state = line.split('=')[1].strip('"')
                    if state != "running":
                        self.logger.error(f"VM {vm_name} is not running (state: {state})")
                        return False
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying VM state: {e}")
            return False

    async def _detect_app_type(self, app_path: str) -> AppType:
        """Detect the type of application"""
        try:
            # Check for key files
            key_files_cmd = f"ssh -p 2222 omar@127.0.0.1 \"(ls {app_path}/*.py 2>/dev/null || echo '') && " \
                           f"(ls {app_path}/requirements.txt 2>/dev/null || echo '') && " \
                           f"(ls {app_path}/package.json 2>/dev/null || echo '') && " \
                           f"(ls {app_path}/pom.xml 2>/dev/null || echo '') && " \
                           f"(ls {app_path}/build.gradle 2>/dev/null || echo '') && " \
                           f"(ls {app_path}/app.py 2>/dev/null || echo '')\""

            key_files = await self._run_command(key_files_cmd)

            if re.search(r'requirements\.txt|\.py$', key_files, re.MULTILINE):
                return AppType.FLASK
            if re.search(r'package\.json', key_files, re.MULTILINE):
                return AppType.NODE
            if re.search(r'pom\.xml|build\.gradle', key_files, re.MULTILINE):
                return AppType.SPRING

            # If we have OpenAI client, use it for detection
            if self.client:
                file_contents = {}
                for file in key_files.strip().split('\n'):
                    if file:
                        try:
                            content = await self._run_command(f"ssh -p 2222 omar@127.0.0.1 'head -20 {file}'")
                            file_contents[os.path.basename(file)] = content
                        except:
                            pass

                file_info = "\n".join([f"File: {k}\nContent sample: {v[:500]}..." for k, v in file_contents.items()])

                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at detecting application types based on key files."},
                        {"role": "user", "content": f"Based on these key files, what type of application is this? Choose from: flask, node, spring, django, react, angular, vue. Files found:\n{file_info}"}
                    ]
                )

                detected_type = response.choices[0].message.content.lower()
                for app_type in AppType:
                    if app_type.value in detected_type:
                        return app_type

            return AppType.UNKNOWN

        except Exception as e:
            self.logger.error(f"Failed to detect app type: {e}")
            return AppType.UNKNOWN

    async def _detect_app_port(self, app_path: str, app_type: AppType) -> int:
        """Detect the port the application uses"""
        try:
            if self.client:
                # Get entrypoint file
                entrypoint = None
                if app_type == AppType.FLASK:
                    candidates = ["app.py", "main.py", "run.py"]
                    for candidate in candidates:
                        try:
                            await self._run_command(f"ssh -p 2222 omar@127.0.0.1 'test -f {app_path}/{candidate}'")
                            entrypoint = f"{app_path}/{candidate}"
                            break
                        except:
                            continue

                if entrypoint:
                    snippet = await self._run_command(f"ssh -p 2222 omar@127.0.0.1 'head -n 200 {entrypoint}'")
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are an expert at reading application code and identifying the port the app will listen on."},
                            {"role": "user", "content": f"Here is the entrypoint file for my {app_type.value} app:\n\n{snippet}\n\nOn which TCP port will this application listen by default?"}
                        ]
                    )
                    answer = response.choices[0].message.content
                    m = re.search(r"\b(\d{2,5})\b", answer)
                    if m:
                        return int(m.group(1))

            # Fallback to conventional defaults
            return {
                AppType.FLASK: 5000,
                AppType.DJANGO: 8000,
                AppType.NODE: 3000,
                AppType.SPRING: 8080,
            }.get(app_type, 8080)

        except Exception as e:
            self.logger.error(f"Failed to detect app port: {e}")
            return 8080

    async def migrate_application(self) -> MigrationState:
        """Execute the application migration"""
        try:
            # Verify VMs are running
            if not await self.verify_vm_running(self.source_vm):
                return MigrationState(
                    source_vm=self.source_vm,
                    target_vm=self.target_vm,
                    application=self.application,
                    port=self.port,
                    status="failed",
                    timestamp=datetime.now().isoformat(),
                    error=f"Source VM {self.source_vm} is not running"
                )
            
            if not await self.verify_vm_running(self.target_vm):
                return MigrationState(
                    source_vm=self.source_vm,
                    target_vm=self.target_vm,
                    application=self.application,
                    port=self.port,
                    status="failed",
                    timestamp=datetime.now().isoformat(),
                    error=f"Target VM {self.target_vm} is not running"
                )

            # Set up port forwarding for SSH
            await self._run_command(f"VBoxManage controlvm {self.source_vm} natpf1 delete migration_ssh", check=False)
            await self._run_command(f"VBoxManage controlvm {self.target_vm} natpf1 delete migration_ssh", check=False)
            await self._run_command(f"VBoxManage controlvm {self.source_vm} natpf1 migration_ssh,tcp,127.0.0.1,2222,,22")
            await self._run_command(f"VBoxManage controlvm {self.target_vm} natpf1 migration_ssh,tcp,127.0.0.1,2223,,22")

            # Detect application type and port
            app_type = await self._detect_app_type(self.application)
            app_port = await self._detect_app_port(self.application, app_type)
            
            # Override with user-provided port if specified
            if self.port:
                app_port = self.port

            # Get app configuration
            config = self.app_configs.get(app_type.value, self.app_configs[AppType.UNKNOWN.value])
            
            # Install dependencies on target VM
            for dep in config.dependencies:
                await self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 "
                    f"'which {dep.split()[0]} || sudo apt-get update && sudo apt-get install -y {dep}'",
                    check=False
                )

            # Copy application files
            with tempfile.TemporaryDirectory() as tmp_dir:
                exclude_args = " ".join([f"--exclude='{pattern}'" for pattern in config.exclude_patterns])
                
                # Copy from source to temp
                await self._run_command(
                    f"rsync -avz -e 'ssh -p 2222' {exclude_args} "
                    f"omar@127.0.0.1:{self.application}/ {tmp_dir}/"
                )
                
                # Copy from temp to target
                await self._run_command(
                    f"rsync -avz -e 'ssh -p 2223' {tmp_dir}/ "
                    f"omar@127.0.0.1:{self.application}/"
                )

            # Run pre-migration commands
            for cmd in config.pre_migration_commands:
                await self._run_command(f"ssh -p 2223 omar@127.0.0.1 'cd {self.application} && {cmd}'", check=False)

            # Set up application service
            service_name = os.path.basename(self.application.rstrip('/')) or "app-service"
            
            if app_type == AppType.FLASK:
                service_content = f"""\
[Unit]
Description=Flask Application Service
After=network.target

[Service]
User=omar
WorkingDirectory={self.application}
ExecStart={self.application}/venv/bin/python {self.application}/app.py
Restart=always
RestartSec=10
Environment="PORT={app_port}"

[Install]
WantedBy=multi-user.target"""
            else:
                service_content = f"""\
[Unit]
Description=Application Service
After=network.target

[Service]
User=omar
WorkingDirectory={self.application}
ExecStart=/bin/bash -c 'cd {self.application} && {" ".join(config.post_migration_commands)}'
Restart=always
RestartSec=10
Environment="PORT={app_port}"

[Install]
WantedBy=multi-user.target"""

            # Create and deploy service file
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(service_content)
                tmp_path = f.name

            await self._run_command(f"scp -P 2223 {tmp_path} omar@127.0.0.1:/tmp/{service_name}.service")
            os.unlink(tmp_path)

            # Enable and start service
            await self._run_command(
                f"ssh -p 2223 omar@127.0.0.1 "
                f"'sudo mv /tmp/{service_name}.service /etc/systemd/system/ && "
                f"sudo systemctl daemon-reload && "
                f"sudo systemctl enable {service_name} && "
                f"sudo systemctl restart {service_name}'",
                check=False
            )

            # Set up port forwarding for application
            await self._run_command(f"VBoxManage controlvm {self.target_vm} natpf1 delete app_port", check=False)
            await self._run_command(
                f"VBoxManage controlvm {self.target_vm} natpf1 app_port,tcp,127.0.0.1,{app_port},,{app_port}",
                check=False
            )

            # Wait for application to start
            await asyncio.sleep(5)

            # Verify application is running
            try:
                check_port = await self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 'sudo lsof -i :{app_port}'",
                    check=False
                )
                if not check_port:
                    raise Exception(f"Application not running on port {app_port}")
            except Exception as e:
                return MigrationState(
                    source_vm=self.source_vm,
                    target_vm=self.target_vm,
                    application=self.application,
                    port=app_port,
                    status="failed",
                    timestamp=datetime.now().isoformat(),
                    error=f"Application verification failed: {str(e)}",
                    app_type=app_type,
                    app_port=app_port
                )

            return MigrationState(
                source_vm=self.source_vm,
                target_vm=self.target_vm,
                application=self.application,
                port=app_port,
                status="completed",
                timestamp=datetime.now().isoformat(),
                app_type=app_type,
                app_port=app_port
            )

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            return MigrationState(
                source_vm=self.source_vm,
                target_vm=self.target_vm,
                application=self.application,
                port=self.port,
                status="failed",
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

async def main():
    parser = argparse.ArgumentParser(description='Migrate application between running VMs')
    parser.add_argument('--source-vm', required=True, help='Source VM name (must be running)')
    parser.add_argument('--target-vm', required=True, help='Target VM name (must be running)')
    parser.add_argument('--application', required=True, help='Application path on the VMs')
    parser.add_argument('--port', type=int, help='Application port')
    parser.add_argument('--health-url', help='URL to verify application state')
    parser.add_argument('--timeout', type=int, default=300, help='Health check timeout in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--openai-api-key', help='OpenAI API key for AI features')
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    migrator = AppMigrator(
        source_vm=args.source_vm,
        target_vm=args.target_vm,
        application=args.application,
        port=args.port,
        health_url=args.health_url,
        timeout=args.timeout,
        debug=args.debug,
        openai_api_key=args.openai_api_key or os.getenv("OPENAI_API_KEY")
    )
    
    result = await migrator.migrate_application()
    
    # Output result in JSON format
    print(json.dumps({
        'source_vm': result.source_vm,
        'target_vm': result.target_vm,
        'application': result.application,
        'port': result.port,
        'status': result.status,
        'timestamp': result.timestamp,
        'error': result.error,
        'app_type': result.app_type.value if result.app_type else None,
        'app_port': result.app_port
    }, indent=2))
    
    sys.exit(0 if result.status == "completed" else 1)

if __name__ == "__main__":
    asyncio.run(main())
