#!/usr/bin/env bash
set -e

pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python download_weights.py
flask db upgrade
