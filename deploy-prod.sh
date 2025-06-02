#!/bin/bash
set -e

# Go to app directory
cd /home/ubuntu/YoloService

# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt

# Move systemd service file and restart
sudo cp yolo-prod.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-prod.service
sudo systemctl restart yolo-prod.service
