#!/bin/bash
# Manual app migration script

SRC_VM="Lubuntu"
SRC_PORT=2223
DEST_VM="ubuntu-24.10"
DEST_PORT=2222
APP_PATH="/opt/sampleapp"
APP_SERVICE="sampleapp"
SSH_KEY="~/.ssh/id_rsa_ansible"

echo "=== Manual App Migration ==="
echo "From: $SRC_VM ($SRC_PORT)"
echo "To: $DEST_VM ($DEST_PORT)"
echo "App: $APP_PATH"

# Step 1: Archive the app on source
echo -e "\n[1] Creating archive on source VM..."
ssh -i $SSH_KEY -p $SRC_PORT omar@127.0.0.1 "sudo tar czf /tmp/sampleapp.tgz -C $APP_PATH ."

if [ $? -ne 0 ]; then
    echo "Failed to create archive on source VM. Make sure the directory exists."
    exit 1
fi

# Step 2: Copy archive to local machine
echo -e "\n[2] Copying archive to local machine..."
scp -i $SSH_KEY -P $SRC_PORT omar@127.0.0.1:/tmp/sampleapp.tgz /tmp/

if [ $? -ne 0 ]; then
    echo "Failed to copy archive from source VM."
    exit 1
fi

# Step 3: Copy archive to destination
echo -e "\n[3] Copying archive to destination VM..."
scp -i $SSH_KEY -P $DEST_PORT /tmp/sampleapp.tgz omar@127.0.0.1:/tmp/

if [ $? -ne 0 ]; then
    echo "Failed to copy archive to destination VM."
    exit 1
fi

# Step 4: Extract archive on destination
echo -e "\n[4] Extracting archive on destination VM..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "sudo mkdir -p $APP_PATH && sudo tar xzf /tmp/sampleapp.tgz -C $APP_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to extract archive on destination VM."
    exit 1
fi

# Step 5: Fix permissions on destination
echo -e "\n[5] Fixing permissions on destination VM..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "sudo chown -R omar:omar $APP_PATH"

if [ $? -ne 0 ]; then
    echo "Failed to fix permissions on destination VM."
    exit 1
fi

# Step 6: Set up virtual environment on destination if needed
echo -e "\n[6] Setting up virtual environment on destination VM..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "cd $APP_PATH && [ -d venv ] || python3 -m venv venv && venv/bin/pip install -r requirements.txt"

if [ $? -ne 0 ]; then
    echo "Failed to set up virtual environment on destination VM."
    exit 1
fi

# Step 7: Create systemd service file
echo -e "\n[7] Creating systemd service file..."
cat > /tmp/sampleapp.service << EOF
[Unit]
Description=Sample Flask Application
After=network.target

[Service]
User=omar
WorkingDirectory=$APP_PATH
ExecStart=$APP_PATH/venv/bin/python $APP_PATH/app.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Step 8: Copy service file to destination
echo -e "\n[8] Copying service file to destination VM..."
scp -i $SSH_KEY -P $DEST_PORT /tmp/sampleapp.service omar@127.0.0.1:/tmp/

if [ $? -ne 0 ]; then
    echo "Failed to copy service file to destination VM."
    exit 1
fi

# Step 9: Install and start service
echo -e "\n[9] Installing and starting service..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "sudo cp /tmp/sampleapp.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable sampleapp && sudo systemctl start sampleapp"

if [ $? -ne 0 ]; then
    echo "Failed to install and start service."
    exit 1
fi

# Step 10: Verify service is running
echo -e "\n[10] Verifying service is running..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "sudo systemctl status sampleapp"

# Step 11: Check if port is listening
echo -e "\n[11] Checking if port is listening..."
ssh -i $SSH_KEY -p $DEST_PORT omar@127.0.0.1 "netstat -tuln | grep 5050 || echo 'Port not listening'"

echo -e "\nMigration completed. You can check if the app is running by accessing http://127.0.0.1:5050"