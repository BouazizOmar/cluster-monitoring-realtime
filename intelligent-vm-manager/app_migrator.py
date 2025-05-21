#!/usr/bin/env python3
import subprocess
import shlex
import time
import os
import json
import logging
import re
from pathlib import Path
import tempfile
import shutil
from typing import Optional, List, Dict, Tuple
from openai import OpenAI
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SSH password for automation
SSH_PASSWORD = "omar"

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


class AppMigrator:
    def __init__(self, log_file: str = "app_migration.log", openai_api_key: Optional[str] = None):
        self.logger = self._setup_logger(log_file)
        # Make OpenAI client optional since we're having issues with it
        try:
            if openai_api_key:
                self.client = OpenAI(api_key=openai_api_key)
            else:
                self.client = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")
            self.client = None
        self.app_configs = self._load_app_configs()

    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("AppMigrator")
        logger.setLevel(logging.INFO)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        handler = logging.FileHandler(log_dir / log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _run_command(self, cmd: str, check: bool = True) -> str:
        self.logger.info(f"Running command: {cmd}")
        try:
            # Add password injection for SSH commands
            if 'ssh ' in cmd or 'scp ' in cmd or 'rsync' in cmd:
                # Only modify commands that require SSH authentication
                if '-i ' not in cmd:  # Skip commands already using SSH key authentication
                    # Replace ssh with sshpass -p password ssh
                    cmd = cmd.replace('ssh ', f'sshpass -p {SSH_PASSWORD} ssh ')
                    cmd = cmd.replace('scp ', f'sshpass -p {SSH_PASSWORD} scp ')
                    # Handle rsync which might be using ssh as a transport
                    if 'rsync' in cmd and '-e \'ssh' in cmd:
                        cmd = cmd.replace('-e \'ssh', f'-e \'sshpass -p {SSH_PASSWORD} ssh')
            
            result = subprocess.run(cmd, shell=True, check=check,
                                    capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.stderr}")
            raise

    def _ensure_vm_running(self, vm_name: str):
        try:
            state = subprocess.check_output([
                "VBoxManage", "showvminfo", vm_name, "--machinereadable"
            ]).decode()

            if "VMState=\"running\"" not in state:
                self.logger.info(f"Starting VM {vm_name}")
                subprocess.run(["VBoxManage", "startvm", vm_name, "--type", "headless"],
                               check=True)
                time.sleep(10)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start VM {vm_name}: {e}")
            raise

    def _create_backup(self, vm_name: str):
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
        try:
            try:
                self.logger.info(f"Removing existing migration_ssh rule for {vm_name}")
                subprocess.run([
                    "VBoxManage", "controlvm", vm_name,
                    "natpf1", "delete", "migration_ssh"
                ], check=False)
                time.sleep(1)
            except subprocess.CalledProcessError:
                pass

            self.logger.info(f"Setting up port forwarding for {vm_name}")
            subprocess.run([
                "VBoxManage", "controlvm", vm_name,
                "natpf1", f"migration_ssh,tcp,127.0.0.1,{host_port},,{guest_port}"
            ], check=True)
            time.sleep(2)

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to set up port forwarding: {e}")
            raise

    def _load_app_configs(self) -> Dict[str, AppConfig]:
        return {
            AppType.FLASK.value: AppConfig(
                app_type=AppType.FLASK,
                exclude_patterns=["venv", "__pycache__", "*.pyc", "*.pyo", "*.pyd"],
                pre_migration_commands=["python3 -m venv venv",
                                        "source venv/bin/activate && pip install -r requirements.txt"],
                post_migration_commands=[],
                health_check_url="/health",
                health_check_port=None,  # Will be detected from code
                dependencies=["python3", "pip", "python3-venv"]
            ),
            AppType.NODE.value: AppConfig(
                app_type=AppType.NODE,
                exclude_patterns=["node_modules", "dist", "build"],
                pre_migration_commands=["npm install"],
                post_migration_commands=["npm start"],
                health_check_url="/api/health",
                health_check_port=None,  # Will be detected from code
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

    def _detect_app_type(self, app_path: str) -> AppType:
        try:
            # Modified to only look at key files instead of entire directory structure
            # This reduces the token count significantly
            key_files_cmd = f"ssh -p 2222 omar@127.0.0.1 \"(ls {app_path}/*.py 2>/dev/null || echo '') && " \
                            f"(ls {app_path}/requirements.txt 2>/dev/null || echo '') && " \
                            f"(ls {app_path}/package.json 2>/dev/null || echo '') && " \
                            f"(ls {app_path}/pom.xml 2>/dev/null || echo '') && " \
                            f"(ls {app_path}/build.gradle 2>/dev/null || echo '') && " \
                            f"(ls {app_path}/app.py 2>/dev/null || echo '')\""

            key_files = self._run_command(key_files_cmd)

            # Check for specific files that indicate app type
            if re.search(r'requirements\.txt|\.py$', key_files, re.MULTILINE):
                self.logger.info("Python application detected based on file extensions and requirements.txt")
                return AppType.FLASK

            if re.search(r'package\.json', key_files, re.MULTILINE):
                self.logger.info("Node.js application detected based on package.json")
                return AppType.NODE

            if re.search(r'pom\.xml|build\.gradle', key_files, re.MULTILINE):
                self.logger.info("Spring application detected based on build files")
                return AppType.SPRING

            # Only if basic detection fails, use AI with limited file data
            if key_files and self.client:
                # Get a sample of content from key files
                file_contents = {}
                for file in key_files.strip().split('\n'):
                    if file:
                        try:
                            # Just get first 20 lines of each file to limit token usage
                            content = self._run_command(f"ssh -p 2222 omar@127.0.0.1 'head -20 {file}'")
                            file_contents[os.path.basename(file)] = content
                        except:
                            pass

                file_info = "\n".join([f"File: {k}\nContent sample: {v[:500]}..." for k, v in file_contents.items()])

                # Limit the file list to reduce token usage
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system",
                             "content": "You are an expert at detecting application types based on key files."},
                            {"role": "user",
                             "content": f"Based on these key files, what type of application is this? Choose from: flask, node, spring, django, react, angular, vue. Files found:\n{file_info}"}
                        ]
                    )

                    detected_type = response.choices[0].message.content.lower()

                    for app_type in AppType:
                        if app_type.value in detected_type:
                            return app_type
                except Exception as e:
                    self.logger.warning(f"OpenAI detection failed: {e}")

            # Fall back to basic detection if we couldn't determine from key files
            if os.path.basename(app_path) == "flask_app" or "flask" in app_path.lower():
                self.logger.info("Detected Flask app based on directory name")
                return AppType.FLASK

            return AppType.UNKNOWN

        except Exception as e:
            self.logger.error(f"Failed to detect app type: {e}")

            # Basic fallback detection based on directory name
            if os.path.basename(app_path) == "flask_app" or "flask" in app_path.lower():
                self.logger.info("Falling back to Flask app based on directory name")
                return AppType.FLASK

            return AppType.UNKNOWN

    def _detect_port_via_ai(self, entrypoint_path: str, app_type: AppType) -> int:
        """Given the path to the app's main file, grab a snippet
        and ask GPT-4 what port the app will run on."""
        # First try to parse the port directly from the file
        try:
            snippet = self._run_command(
                f"ssh -p 2222 omar@127.0.0.1 \"head -n 200 {entrypoint_path} || cat {entrypoint_path}\""
            )
            
            # Look for common patterns in Flask apps
            port_patterns = [
                r'app\.run\(.*port\s*=\s*(\d+)',
                r'--port\s*[=\s](\d+)',
                r'PORT\s*[=:]\s*(\d+)'
            ]
            
            for pattern in port_patterns:
                match = re.search(pattern, snippet)
                if match:
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        pass
        except Exception as e:
            self.logger.warning(f"Failed to parse port from file: {e}")
        
        # If direct parsing failed and we have OpenAI client, try AI
        if self.client:
            try:
                # 2. Craft a prompt that covers multiple frameworks
                system = {
                    "role": "system",
                    "content": (
                        "You are an expert at reading application code and "
                        "identifying on which port the app will listen by default. "
                        "The app may be Flask (Python), Express (Node.js), "
                        "Spring Boot (Java), Django, or similar."
                    )
                }
                user = {
                    "role": "user",
                    "content": (
                        f"Here is the entrypoint file for my {app_type.value} app:\n\n"
                        f"{snippet}\n\n"
                        "Question: On which TCP port will this application listen by default?"
                    )
                }
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system, user]
                )
                answer = resp.choices[0].message.content
                # 3. Extract the first integer from the answer
                m = re.search(r"\b(\d{2,5})\b", answer)
                if m:
                    return int(m.group(1))
            except Exception as e:
                self.logger.warning(f"OpenAI port detection failed: {e}")
        
        # 4. Fallback to conventional defaults
        return {
            AppType.FLASK: 5000,
            AppType.DJANGO: 8000,
            AppType.NODE: 3000,
            AppType.SPRING: 8080,
        }.get(app_type, 8080)

    def _detect_app_port_and_entrypoint(self, app_path: str, app_type: AppType) -> Tuple[int, str]:
        """Universal detection: find your main file, then ask AI for port."""
        # 1) Identify entrypoint
        if app_type == AppType.FLASK or app_type == AppType.DJANGO:
            candidates = ["app.py", "manage.py", "main.py", "run.py"]
            glob = "*.py"
        elif app_type == AppType.NODE:
            candidates = []
            glob = "package.json"
        elif app_type == AppType.SPRING:
            candidates = ["Application.java"]
            glob = "pom.xml"
        else:
            candidates = []
            glob = "*"
        entrypoint = None
        # quick heuristic: look for named files first
        for fn in candidates:
            full = os.path.join(app_path, fn)
            try:
                self._run_command(f"ssh -p 2222 omar@127.0.0.1 'test -f {full}'")
                entrypoint = full
                break
            except:
                pass
        # fallback: pick the first matching file if no candidate found
        if not entrypoint:
            out = self._run_command(
                f"ssh -p 2222 omar@127.0.0.1 \"find {app_path} -maxdepth 1 -type f -name '{glob}'\""
            ).strip().splitlines()
            if out:
                entrypoint = out[0]
        # 2) Ask AI for the port
        port = self._detect_port_via_ai(entrypoint, app_type) if entrypoint else None
        # 3) Fallback defaults if AI fails or no entrypoint
        if not port:
            port = {
                AppType.FLASK: 5000,
                AppType.DJANGO: 8000,
                AppType.NODE: 3000,
                AppType.SPRING: 8080,
            }.get(app_type, 8080)
        self.logger.info(f"Detected app port: {port}, entrypoint: {entrypoint}")
        return port, entrypoint

    def _analyze_dependencies(self, app_path: str) -> List[str]:
        try:
            # Simplified dependency detection for Python projects
            has_requirements = self._run_command(
                f"ssh -p 2222 omar@127.0.0.1 'test -f {app_path}/requirements.txt && echo yes || echo no'").strip()

            if has_requirements == "yes":
                return ["python3", "pip", "python3-venv"]

            # Check for package.json
            has_package_json = self._run_command(
                f"ssh -p 2222 omar@127.0.0.1 'test -f {app_path}/package.json && echo yes || echo no'").strip()

            if has_package_json == "yes":
                return ["nodejs", "npm"]

            # Default dependencies
            return []

        except Exception as e:
            self.logger.error(f"Failed to analyze dependencies: {e}")
            return []

    def _generate_migration_plan(self, app_path: str, detected_type: AppType) -> AppConfig:
        # Skip AI migration plan generation to avoid token issues
        self.logger.info(f"Using default configuration for {detected_type.value}")

        if detected_type.value in self.app_configs:
            return self.app_configs[detected_type.value]
        else:
            return self.app_configs[AppType.UNKNOWN.value]

    def migrate_app(self,
                    source_vm: str,
                    target_vm: str,
                    app_path: str = "/opt/sampleapp",
                    app_type: Optional[str] = None,
                    exclude_patterns: Optional[List[str]] = None,
                    pre_migration_commands: Optional[List[str]] = None,
                    post_migration_commands: Optional[List[str]] = None,
                    health_check_url: Optional[str] = None,
                    health_check_port: Optional[int] = None,
                    local_app_path: Optional[str] = None):
        try:
            # First check if sshpass is installed
            try:
                subprocess.run("which sshpass", shell=True, check=True, stdout=subprocess.PIPE)
            except subprocess.CalledProcessError:
                self.logger.warning("sshpass is not installed. Trying to install it...")
                try:
                    subprocess.run("brew install sshpass || sudo apt-get install -y sshpass || sudo yum install -y sshpass", 
                                  shell=True, check=False)
                except Exception as e:
                    self.logger.warning(f"Could not install sshpass: {e}. SSH operations may require password entry.")
            
            self.logger.info(f"Starting migration from {source_vm} to {target_vm}")

            # 1. Ensure both VMs are running
            self._ensure_vm_running(source_vm)
            self._ensure_vm_running(target_vm)

            # 2. Set up port forwarding
            self._setup_port_forwarding(source_vm, 2222, 22)
            self._setup_port_forwarding(target_vm, 2223, 22)

            # 3. Create backup
            backup_name = self._create_backup(source_vm)

            # 4. Copy local app if provided
            if local_app_path:
                self.logger.info(f"Copying local application from {local_app_path} to {source_vm}")
                self._run_command(f"ssh -p 2222 omar@127.0.0.1 'mkdir -p {app_path}'", check=False)
                self._run_command(
                    f"rsync -avz {local_app_path}/ -e 'ssh -p 2222' "
                    f"omar@127.0.0.1:{app_path}/"
                )

            # 5. Detect application type or use provided type
            if app_type:
                detected_type = AppType(app_type)
                self.logger.info(f"Using provided application type: {detected_type.value}")
            else:
                detected_type = self._detect_app_type(app_path)
                self.logger.info(f"Detected application type: {detected_type.value}")

            # 6. Detect app port and entrypoint
            app_port, entrypoint = self._detect_app_port_and_entrypoint(app_path, detected_type)

            # Override with user-provided port if specified
            if health_check_port:
                app_port = health_check_port
                self.logger.info(f"Using provided health check port: {app_port}")

            # 7. Load configuration and update with detected values
            config = self.app_configs[detected_type.value]
            config.health_check_port = app_port
            config.app_entrypoint = entrypoint

            # Update with user-provided values if specified
            if exclude_patterns:
                config.exclude_patterns = exclude_patterns
            if pre_migration_commands:
                config.pre_migration_commands = pre_migration_commands
            if post_migration_commands:
                config.post_migration_commands = post_migration_commands
            if health_check_url:
                config.health_check_url = health_check_url

            # 8. Install dependencies
            self.logger.info(f"Installing dependencies: {config.dependencies}")
            for dep in config.dependencies:
                self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 "
                    f"'which {dep.split()[0]} || sudo apt-get update && sudo apt-get install -y {dep}'",
                    check=False
                )

            # 9. Copy application files
            self._run_command(f"ssh -p 2223 omar@127.0.0.1 'mkdir -p {app_path}'", check=False)
            with tempfile.TemporaryDirectory() as tmp_dir:
                exclude_args = " ".join([f"--exclude='{pattern}'" for pattern in config.exclude_patterns])

                self.logger.info("Copying application files from source VM")
                self._run_command(
                    f"rsync -avz -e 'ssh -p 2222' {exclude_args} "
                    f"omar@127.0.0.1:{app_path}/ {tmp_dir}/"
                )

                self.logger.info("Copying application files to target VM")
                self._run_command(
                    f"rsync -avz -e 'ssh -p 2223' {tmp_dir}/ "
                    f"omar@127.0.0.1:{app_path}/"
                )

            # 10. App-specific setup
            if config.app_type == AppType.FLASK:
                self.logger.info("Configuring Flask application")

                # Create virtual environment
                self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 "
                    f"'cd {app_path} && python3 -m venv venv && "
                    f"venv/bin/pip install -r requirements.txt'",
                    check=False
                )

                # Determine app entrypoint filename
                entrypoint_filename = os.path.basename(config.app_entrypoint) if config.app_entrypoint else "app.py"

                # Configure systemd service
                service_name = os.path.basename(app_path.rstrip('/')) or "flask-app"
                service_content = f"""\
[Unit]
Description=Flask Application Service
After=network.target

[Service]
User=omar
WorkingDirectory={app_path}
ExecStart={app_path}/venv/bin/python {app_path}/{entrypoint_filename}
Restart=always
RestartSec=10
Environment="PORT={app_port}"
Environment="FLASK_APP={entrypoint_filename}"
Environment="FLASK_ENV=development"
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target"""

                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(service_content)
                    tmp_path = f.name

                self._run_command(f"scp -P 2223 {tmp_path} omar@127.0.0.1:/tmp/{service_name}.service")
                os.unlink(tmp_path)

                # Enable and start service with improved error handling
                try:
                    # First, try to use sudo without expecting a password prompt
                    self._run_command(
                        f"ssh -p 2223 omar@127.0.0.1 "
                        f"'sudo -n mv /tmp/{service_name}.service /etc/systemd/system/ && "
                        f"sudo -n systemctl daemon-reload && "
                        f"sudo -n systemctl enable {service_name} && "
                        f"sudo -n systemctl restart {service_name}'",
                        check=False
                    )
                    
                    # Check service status
                    status = self._run_command(
                        f"ssh -p 2223 omar@127.0.0.1 'sudo -n systemctl status {service_name}'",
                        check=False
                    )
                    self.logger.info(f"Service status:\n{status}")
                    
                    # If service failed to start properly, try a direct run approach
                    if "Active: active (running)" not in status:
                        self.logger.warning("Service not running properly. Trying direct run method.")
                        self._run_direct_app_start(app_path, entrypoint_filename, app_port)
                except Exception as e:
                    self.logger.warning(f"Error starting service: {e}. Trying direct run method.")
                    self._run_direct_app_start(app_path, entrypoint_filename, app_port)

            elif config.app_type == AppType.NODE:
                self.logger.info("Configuring Node.js application")

                # Install dependencies
                self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 "
                    f"'cd {app_path} && npm install'",
                    check=False
                )

                # Determine entrypoint file
                entrypoint_filename = os.path.basename(config.app_entrypoint) if config.app_entrypoint else "index.js"

                # Configure systemd service
                service_name = os.path.basename(app_path.rstrip('/')) or "node-app"
                service_content = f"""\
[Unit]
Description=Node.js Application Service
After=network.target

[Service]
User=omar
WorkingDirectory={app_path}
ExecStart=/usr/bin/node {app_path}/{entrypoint_filename}
Restart=always
RestartSec=10
Environment="PORT={app_port}"

[Install]
WantedBy=multi-user.target"""

                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(service_content)
                    tmp_path = f.name

                self._run_command(f"scp -P 2223 {tmp_path} omar@127.0.0.1:/tmp/{service_name}.service")
                os.unlink(tmp_path)

                # Enable and start service
                self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 "
                    f"'sudo mv /tmp/{service_name}.service /etc/systemd/system/ && "
                    f"sudo systemctl daemon-reload && "
                    f"sudo systemctl enable {service_name} && "
                    f"sudo systemctl restart {service_name}'",
                    check=False
                )

                # Verify service status
                status = self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 'sudo systemctl status {service_name}'",
                    check=False
                )
                self.logger.info(f"Service status:\n{status}")

            # 11. Configure port forwarding for application port
            self.logger.info(f"Setting up port forwarding on port {app_port}")
            self._run_command(f"VBoxManage controlvm {target_vm} natpf1 delete app_port", check=False)
            self._run_command(
                f"VBoxManage controlvm {target_vm} natpf1 app_port,tcp,127.0.0.1,{app_port},,{app_port}",
                check=False
            )

            # 12. Wait for app to be ready and verify it's running
            self.logger.info(f"Waiting for application to start up...")
            time.sleep(5)  # Give the app time to start

            try:
                # Check if something is listening on the port
                check_port = self._run_command(
                    f"ssh -p 2223 omar@127.0.0.1 'sudo lsof -i :{app_port}'",
                    check=False
                )

                if check_port:
                    self.logger.info(f"Application is running on port {app_port}")
                else:
                    self.logger.warning(f"Could not verify if application is running on port {app_port}")
            except:
                self.logger.warning("Failed to check if application is running")

            self.logger.info("Migration completed successfully!")
            self.logger.info(f"Application available at http://127.0.0.1:{app_port}")
            return True, app_port

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            if 'backup_name' in locals():
                try:
                    subprocess.run(["VBoxManage", "controlvm", source_vm, "poweroff"], check=True)
                    time.sleep(5)
                    subprocess.run(["VBoxManage", "snapshot", source_vm, "restore", backup_name], check=True)
                    self.logger.info("Rollback successful")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Rollback failed: {e}")
            return False, None

    def _run_direct_app_start(self, app_path: str, entrypoint_filename: str, app_port: int):
        """Run the Flask app directly via SSH in the background as a fallback method."""
        self.logger.info(f"Starting app directly on port {app_port}")
        
        # Create a startup script
        startup_script = f"""#!/bin/bash
cd {app_path}
source venv/bin/activate
export PORT={app_port}
export FLASK_APP={entrypoint_filename}
export FLASK_ENV=development
python -m flask run --host=0.0.0.0 --port={app_port} > {app_path}/app.log 2>&1 &
echo $! > {app_path}/app.pid
"""
        
        # Create and copy the startup script
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(startup_script)
            script_path = f.name
            
        self._run_command(f"scp -P 2223 {script_path} omar@127.0.0.1:/tmp/start_flask.sh")
        os.unlink(script_path)
        
        # Make it executable and run it
        self._run_command(
            f"ssh -p 2223 omar@127.0.0.1 "
            f"'chmod +x /tmp/start_flask.sh && /tmp/start_flask.sh'",
            check=False
        )
        
        # Wait a moment for the app to start
        time.sleep(3)
        
        # Verify it's running
        ps_output = self._run_command(
            f"ssh -p 2223 omar@127.0.0.1 'ps aux | grep flask'",
            check=False
        )
        self.logger.info(f"App process check: {ps_output}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI-Powered Application Migrator")
    parser.add_argument("source_vm", help="Source VM name")
    parser.add_argument("target_vm", help="Target VM name")
    parser.add_argument("--app-path", default="/opt/sampleapp",
                        help="Path to the application on the VMs")
    parser.add_argument("--local-app-path",
                        help="Path to the application on your local machine")
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

    # Don't use OpenAI for now to avoid client initialization issues
    # openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    openai_api_key = None

    migrator = AppMigrator(openai_api_key=openai_api_key)
    success, port = migrator.migrate_app(
        source_vm=args.source_vm,
        target_vm=args.target_vm,
        app_path=args.app_path,
        app_type=args.app_type,
        exclude_patterns=args.exclude,
        pre_migration_commands=args.pre_migration,
        post_migration_commands=args.post_migration,
        health_check_url=args.health_check_url,
        health_check_port=args.health_check_port,
        local_app_path=args.local_app_path
    )

    if success:
        print(f"Migration successful! Application running on port {port}")
    else:
        print("Migration failed. Check logs for details.")

if __name__ == "__main__":
    main()