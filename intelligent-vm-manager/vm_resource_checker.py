#!/usr/bin/env python3

import os
import sys
import json
import logging
import asyncio
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VMResourceState:
    vm_name: str
    cpus: int
    memory_mb: int
    cpu_usage_percent: float
    memory_usage_mb: int
    state: str
    timestamp: str

class VMResourceChecker:
    def __init__(self, debug: bool = False):
        self.logger = self._setup_logging(debug)
    
    def _setup_logging(self, debug: bool) -> logging.Logger:
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def _run_vboxmanage(self, *args) -> tuple[str, str, int]:
        """Run VBoxManage command and return stdout, stderr, and return code"""
        cmd = ["VBoxManage"] + list(args)
        self.logger.debug(f"Running command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        return stdout.decode(), stderr.decode(), process.returncode
    
    async def get_vm_state(self, vm_name: str) -> Optional[str]:
        """Get current state of VM (running, powered off, etc.)"""
        stdout, stderr, return_code = await self._run_vboxmanage("showvminfo", vm_name, "--machinereadable")
        
        if return_code != 0:
            self.logger.error(f"Failed to get VM state: {stderr}")
            return None
        
        # Parse VM state from output
        for line in stdout.split('\n'):
            if line.startswith('VMState='):
                return line.split('=')[1].strip('"')
        return None
    
    async def get_vm_resources(self, vm_name: str) -> Optional[Dict[str, Any]]:
        """Get current resource allocation for VM"""
        stdout, stderr, return_code = await self._run_vboxmanage("showvminfo", vm_name, "--machinereadable")
        
        if return_code != 0:
            self.logger.error(f"Failed to get VM resources: {stderr}")
            return None
        
        resources = {}
        for line in stdout.split('\n'):
            if line.startswith('cpus='):
                resources['cpus'] = int(line.split('=')[1].strip('"'))
            elif line.startswith('memory='):
                resources['memory_mb'] = int(line.split('=')[1].strip('"'))
        
        return resources if resources else None
    
    async def get_vm_metrics(self, vm_name: str) -> Optional[Dict[str, Any]]:
        """Get current resource usage metrics for VM"""
        stdout, stderr, return_code = await self._run_vboxmanage("metrics", "collect", vm_name)
        
        if return_code != 0:
            self.logger.error(f"Failed to get VM metrics: {stderr}")
            return None
        
        metrics = {}
        for line in stdout.split('\n'):
            if 'CPU/Usage/User' in line:
                metrics['cpu_usage_percent'] = float(line.split('=')[1].strip())
            elif 'RAM/Usage/Total' in line:
                # Convert bytes to MB
                metrics['memory_usage_mb'] = int(float(line.split('=')[1].strip()) / (1024 * 1024))
        
        return metrics if metrics else None
    
    async def check_vm_resources(self, vm_name: str) -> Optional[VMResourceState]:
        """Get complete resource state for a VM"""
        try:
            # Get VM state
            state = await self.get_vm_state(vm_name)
            if not state:
                return None
            
            # Get resource allocation
            resources = await self.get_vm_resources(vm_name)
            if not resources:
                return None
            
            # Initialize metrics with defaults
            metrics = {
                'cpu_usage_percent': 0.0,
                'memory_usage_mb': 0
            }
            
            # Only get metrics if VM is running
            if state == 'running':
                vm_metrics = await self.get_vm_metrics(vm_name)
                if vm_metrics:
                    metrics.update(vm_metrics)
            
            return VMResourceState(
                vm_name=vm_name,
                cpus=resources['cpus'],
                memory_mb=resources['memory_mb'],
                cpu_usage_percent=metrics['cpu_usage_percent'],
                memory_usage_mb=metrics['memory_usage_mb'],
                state=state,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error checking VM resources: {e}")
            return None

async def main():
    parser = argparse.ArgumentParser(description='Check VM resource state')
    parser.add_argument('vm_name', help='Name of the VM to check')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    checker = VMResourceChecker(debug=args.debug)
    resource_state = await checker.check_vm_resources(args.vm_name)
    
    if resource_state:
        if args.json:
            print(json.dumps({
                'vm_name': resource_state.vm_name,
                'cpus': resource_state.cpus,
                'memory_mb': resource_state.memory_mb,
                'cpu_usage_percent': resource_state.cpu_usage_percent,
                'memory_usage_mb': resource_state.memory_usage_mb,
                'state': resource_state.state,
                'timestamp': resource_state.timestamp
            }, indent=2))
        else:
            print(f"\nVM Resource State for {resource_state.vm_name}:")
            print(f"State: {resource_state.state}")
            print(f"CPUs Allocated: {resource_state.cpus}")
            print(f"Memory Allocated: {resource_state.memory_mb} MB")
            print(f"CPU Usage: {resource_state.cpu_usage_percent:.1f}%")
            print(f"Memory Usage: {resource_state.memory_usage_mb} MB")
            print(f"Timestamp: {resource_state.timestamp}")
    else:
        print(f"Failed to get resource state for VM: {args.vm_name}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 