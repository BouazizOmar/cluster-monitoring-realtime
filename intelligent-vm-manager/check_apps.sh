#!/bin/bash
# A simple direct approach to resize a VirtualBox VM
# This uses a more straightforward method to avoid save state issues

# Usage: ./resize_vm.sh VM_NAME CPU_COUNT MEMORY_MB

VM_NAME="$1"
CPU_COUNT="$2"
MEMORY_MB="$3"

# Validate inputs
if [ -z "$VM_NAME" ] || [ -z "$CPU_COUNT" ]; then
  echo "Usage: $0 VM_NAME CPU_COUNT [MEMORY_MB]"
  echo "Example: $0 Lubuntu 4 8192"
  exit 1
fi

# Function to log with timestamp
log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Create snapshot for safety
log "Creating snapshot..."
VBoxManage snapshot "$VM_NAME" take "resize_backup_$(date +%s)" || {
  log "ERROR: Failed to create snapshot"
  exit 1
}

# Check if VM is running
VM_STATE=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "VMState=" | cut -d '"' -f 2)
log "Current VM state: $VM_STATE"

# Get current config
CURRENT_CPU=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "^cpus=" | cut -d '=' -f 2 | tr -d '"')
CURRENT_MEM=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "^memory=" | cut -d '=' -f 2 | tr -d '"')

log "Current CPU: $CURRENT_CPU, Target CPU: $CPU_COUNT"
log "Current Memory: $CURRENT_MEM MB, Target Memory: $MEMORY_MB MB"

# Stop VM if running
if [ "$VM_STATE" = "running" ]; then
  log "VM is running, saving state..."
  # First try save state to preserve apps
  VBoxManage controlvm "$VM_NAME" savestate || {
    log "WARNING: Failed to save state, forcing shutdown"
    VBoxManage controlvm "$VM_NAME" poweroff
  }
  sleep 5
fi

# Ensure VM is truly powered off
log "Ensuring VM is powered off..."
VM_STATE=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "VMState=" | cut -d '"' -f 2)
if [ "$VM_STATE" = "saved" ]; then
  log "VM is in saved state, cannot modify while saved. Powering off..."
  VBoxManage controlvm "$VM_NAME" poweroff
  sleep 5
fi

# Do the actual modification
MODIFY_CMD="VBoxManage modifyvm \"$VM_NAME\""

if [ "$CPU_COUNT" != "$CURRENT_CPU" ]; then
  MODIFY_CMD="$MODIFY_CMD --cpus $CPU_COUNT"
fi

if [ -n "$MEMORY_MB" ] && [ "$MEMORY_MB" != "$CURRENT_MEM" ]; then
  MODIFY_CMD="$MODIFY_CMD --memory $MEMORY_MB"
fi

if [ "$MODIFY_CMD" != "VBoxManage modifyvm \"$VM_NAME\"" ]; then
  log "Executing: $MODIFY_CMD"
  eval "$MODIFY_CMD" || {
    log "ERROR: Failed to modify VM resources"
    exit 1
  }
fi

# Start VM back up
log "Starting VM back up..."
VBoxManage startvm "$VM_NAME" --type headless || {
  log "ERROR: Failed to start VM"
  exit 1
}

# Verify changes
sleep 5
NEW_CPU=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "^cpus=" | cut -d '=' -f 2 | tr -d '"')
NEW_MEM=$(VBoxManage showvminfo "$VM_NAME" --machinereadable | grep "^memory=" | cut -d '=' -f 2 | tr -d '"')

log "Resize complete. New CPU: $NEW_CPU, New Memory: $NEW_MEM MB"
log "Note: Your Flask app will need to be manually restarted"

# Instructions for manual restart after resize
echo "------------------------------------------------------------"
echo "To restart your Flask app, SSH into your VM and run:"
echo "cd ~/Desktop/my-flask-app"
echo "source venv/bin/activate"
echo "python3 app.py"
echo "------------------------------------------------------------"