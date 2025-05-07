#!/usr/bin/env python3
"""
Debug script to check VM connectivity and app status.
Run this script to diagnose issues with your app migration.
"""

import subprocess
import sys
import os
import time
import argparse


def run_command(cmd, timeout=30):
    """Run a command and return output"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False, "Timeout"
    except Exception as e:
        print(f"Error running command: {str(e)}")
        return False, str(e)


def check_ssh_connectivity(host, port, user):
    """Check if we can SSH to the host"""
    print(f"\n--- Checking SSH connectivity to {host}:{port} ---")
    success, output = run_command(
        ["ssh", "-p", str(port), "-o", "ConnectTimeout=5", f"{user}@{host}", "echo 'SSH connection successful'"])
    return success


def check_service_status(host, port, user, service):
    """Check service status on remote host"""
    print(f"\n--- Checking service status for {service} ---")
    success, output = run_command(
        ["ssh", "-p", str(port), f"{user}@{host}", f"systemctl status {service} || echo 'Service not found'"])
    if success:
        print(output)
    return success


def check_app_directory(host, port, user, path):
    """Check if app directory exists and list contents"""
    print(f"\n--- Checking app directory {path} ---")
    success, output = run_command(
        ["ssh", "-p", str(port), f"{user}@{host}", f"ls -la {path} || echo 'Directory not found'"])
    if success:
        print(output)
    return success


def check_port_listening(host, port, user, app_port):
    """Check if app port is listening"""
    print(f"\n--- Checking if port {app_port} is listening ---")
    success, output = run_command(
        ["ssh", "-p", str(port), f"{user}@{host}", f"netstat -tuln | grep {app_port} || echo 'Port not listening'"])
    if success:
        print(output)
    return success


def check_logs(host, port, user, service):
    """Check service logs"""
    print(f"\n--- Checking logs for {service} ---")
    success, output = run_command(
        ["ssh", "-p", str(port), f"{user}@{host}", f"journalctl -u {service} -n 50 --no-pager || echo 'No logs found'"])
    if success:
        print(output)
    return success


def check_file_permissions(host, port, user, path):
    """Check file permissions"""
    print(f"\n--- Checking file permissions in {path} ---")
    success, output = run_command(["ssh", "-p", str(port), f"{user}@{host}",
                                   f"find {path} -type f -name '*.py' -exec ls -l {{}} \\; || echo 'No Python files found'"])
    if success:
        print(output)
    return success


def main():
    parser = argparse.ArgumentParser(description='Debug VM app migration')
    parser.add_argument('--src', help='Source VM name (from inventory)', default='Lubuntu')
    parser.add_argument('--dest', help='Destination VM name (from inventory)', default='ubuntu-24.10')
    parser.add_argument('--app-path', help='Application path', default='/opt/sampleapp')
    parser.add_argument('--service', help='Service name', default='sampleapp')
    parser.add_argument('--port', help='App port', type=int, default=5050)
    args = parser.parse_args()

    # Read inventory file to get connection details
    inventory = {}
    try:
        with open('inventory.ini', 'r') as f:
            for line in f:
                if 'ansible_host' in line and 'ansible_port' in line:
                    parts = line.strip().split()
                    vm_name = parts[0]
                    host = parts[1].split('=')[1]
                    port = parts[2].split('=')[1]
                    inventory[vm_name] = {'host': host, 'port': port}
    except Exception as e:
        print(f"Error reading inventory: {str(e)}")
        sys.exit(1)

    user = os.environ.get('USER', 'omar')

    # Check VM availability
    for vm in [args.src, args.dest]:
        if vm not in inventory:
            print(f"VM {vm} not found in inventory")
            continue

        vm_info = inventory[vm]
        print(f"\n=== Checking VM: {vm} ({vm_info['host']}:{vm_info['port']}) ===")

        # Check SSH
        if not check_ssh_connectivity(vm_info['host'], vm_info['port'], user):
            print(f"Cannot connect to {vm} via SSH. Make sure the VM is running.")
            continue

        # For source VM
        if vm == args.src:
            check_app_directory(vm_info['host'], vm_info['port'], user, args.app_path)
            check_port_listening(vm_info['host'], vm_info['port'], user, args.port)

        # For destination VM
        if vm == args.dest:
            check_app_directory(vm_info['host'], vm_info['port'], user, args.app_path)
            check_service_status(vm_info['host'], vm_info['port'], user, args.service)
            check_port_listening(vm_info['host'], vm_info['port'], user, args.port)
            check_logs(vm_info['host'], vm_info['port'], user, args.service)
            check_file_permissions(vm_info['host'], vm_info['port'], user, args.app_path)

    print("\n=== Debug completed ===")


if __name__ == '__main__':
    main()