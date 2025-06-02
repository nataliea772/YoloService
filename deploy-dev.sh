#!/bin/bash
set -e

cd /home/ubuntu/YoloService

python3 -m venv .venv
source .venv/bin/activate
pip install -r torch-requirements.txt
pip install -r requirements.txt

sudo cp yolo-dev.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable yolo-dev.service
sudo systemctl restart yolo-dev.service
