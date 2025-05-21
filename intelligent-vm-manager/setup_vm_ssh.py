#!/usr/bin/env python3

import os
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, shell=True):
    """Run a shell command and return its output."""
    try:
        result = subprocess.run(cmd, shell=shell, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def setup_vm_ssh(vm_name, vm_ip, ssh_port=22):
    """Set up SSH configuration for a VM."""
    script_dir = Path(__file__).parent
    key_path = script_dir / "key"
    key_pub_path = script_dir / "key.pub"
    
    if not key_path.exists() or not key_pub_path.exists():
        print("SSH keys not found. Generating new keys...")
        run_command(f"ssh-keygen -t ed25519 -f {key_path} -N ''")
    
    # Read the public key
    with open(key_pub_path, 'r') as f:
        public_key = f.read().strip()
    
    # Create .ssh directory and authorized_keys file on the VM
    ssh_setup_commands = [
        f"mkdir -p ~/.ssh",
        f"chmod 700 ~/.ssh",
        f"echo '{public_key}' >> ~/.ssh/authorized_keys",
        f"chmod 600 ~/.ssh/authorized_keys"
    ]
    
    # Execute commands on the VM using password authentication first
    print(f"Setting up SSH for {vm_name}...")
    for cmd in ssh_setup_commands:
        run_command(f"ssh -p {ssh_port} -o StrictHostKeyChecking=no ubuntu@{vm_ip} '{cmd}'")
    
    # Test SSH connection
    print(f"Testing SSH connection to {vm_name}...")
    try:
        run_command(f"ssh -i {key_path} -p {ssh_port} -o StrictHostKeyChecking=no ubuntu@{vm_ip} 'echo SSH connection successful'")
        print(f"SSH setup completed successfully for {vm_name}")
    except subprocess.CalledProcessError:
        print(f"Failed to establish SSH connection to {vm_name}")
        sys.exit(1)

def setup_port_forwarding(vm_name, vm_ip, local_port, vm_port=22):
    """Set up port forwarding for a VM."""
    # Check if port forwarding is already set up
    try:
        result = subprocess.run(
            f"lsof -i :{local_port}",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Port {local_port} is already in use. Please choose a different port.")
            return False
    except subprocess.CalledProcessError:
        pass

    # Set up port forwarding
    print(f"Setting up port forwarding for {vm_name}...")
    try:
        # Kill any existing port forwarding for this VM
        subprocess.run(f"pkill -f 'ssh.*{local_port}:localhost:{vm_port}'", shell=True)
        time.sleep(1)
        
        # Start new port forwarding
        cmd = f"ssh -i {Path(__file__).parent}/key -N -L {local_port}:localhost:{vm_port} ubuntu@{vm_ip} &"
        subprocess.Popen(cmd, shell=True)
        print(f"Port forwarding set up: localhost:{local_port} -> {vm_name}:{vm_port}")
        return True
    except Exception as e:
        print(f"Failed to set up port forwarding: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python setup_vm_ssh.py <vm_name> <vm_ip> [local_port]")
        sys.exit(1)
    
    vm_name = sys.argv[1]
    vm_ip = sys.argv[2]
    local_port = int(sys.argv[3]) if len(sys.argv) > 3 else 2222
    
    setup_vm_ssh(vm_name, vm_ip)
    setup_port_forwarding(vm_name, vm_ip, local_port) 