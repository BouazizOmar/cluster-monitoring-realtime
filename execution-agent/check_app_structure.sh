#!/bin/bash
# Script to check Flask app structure

SRC_VM="Lubuntu"
SRC_PORT=2223
APP_PATH="/opt/sampleapp"
SSH_KEY="~/.ssh/id_rsa_ansible"

echo "=== Checking Flask App Structure ==="
echo "VM: $SRC_VM ($SRC_PORT)"
echo "App path: $APP_PATH"

# Check if the app directory exists
echo -e "\n[1] Checking if app directory exists..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "ls -la $APP_PATH"

# Check Python version
echo -e "\n[2] Checking Python version..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "python3 --version"

# Check app.py contents
echo -e "\n[3] Checking app.py contents..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "cat $APP_PATH/app.py"

# Check requirements.txt
echo -e "\n[4] Checking requirements.txt..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "cat $APP_PATH/requirements.txt"

# Check if venv exists
echo -e "\n[5] Checking if venv exists..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "ls -la $APP_PATH/venv"

# Check currently running process
echo -e "\n[6] Checking running process..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "ps aux | grep python"

# Check port binding
echo -e "\n[7] Checking port binding..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "netstat -tuln | grep 5050"

echo -e "\nApp structure check completed."