#!/usr/bin/env bash
set -euxo pipefail

HF_TOKEN="${1:-${HF_TOKEN:-}}"
HF_REPO="${2:-${HF_REPO:-Tim-Pinecone/LOC-Chronicling-America}}"

if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: Hugging Face token is required!"
    echo "Usage: sudo bash scripts/setup_worker.sh <HF_TOKEN> [HF_REPO]"
    exit 1
fi

APP_DIR="/home/ubuntu/loc-chronicling-america"
git config --global --add safe.directory "${APP_DIR}" || true

echo "=== 1. Writing environment file ==="
printf "HF_TOKEN=%s\nHF_REPO=%s\nPYTHONUNBUFFERED=1\n" "${HF_TOKEN}" "${HF_REPO}" > "${APP_DIR}/.env"
chown ubuntu:ubuntu "${APP_DIR}/.env"
chmod 600 "${APP_DIR}/.env"

echo "=== 2. Creating log file ==="
touch /var/log/chronam-pipeline.log
chown ubuntu:ubuntu /var/log/chronam-pipeline.log

echo "=== 3. Configuring CloudWatch Agent ==="
mkdir -p /opt/aws/amazon-cloudwatch-agent/etc
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << 'EOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/chronam-pipeline.log",
            "log_group_name": "/aws/ec2/loc-chronicling-america",
            "log_stream_name": "{instance_id}/pipeline.log",
            "timezone": "UTC"
          }
        ]
      }
    }
  }
}
EOF

if [ -f /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl ]; then
    /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json || true
fi

echo "=== 4. Creating systemd service ==="
cat > /etc/systemd/system/loc-pipeline.service << 'EOF'
[Unit]
Description=Library of Congress Chronicling America Nationwide Streaming Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/loc-chronicling-america
EnvironmentFile=/home/ubuntu/loc-chronicling-america/.env
ExecStart=/home/ubuntu/loc-chronicling-america/.venv/bin/python -u scripts/run_multi_state_pipeline.py --high-value-36h --purge-local-after-upload
Restart=always
RestartSec=15
StandardOutput=append:/var/log/chronam-pipeline.log
StandardError=append:/var/log/chronam-pipeline.log

[Install]
WantedBy=multi-user.target
EOF

echo "=== 5. Starting service ==="
systemctl daemon-reload
systemctl enable loc-pipeline.service
systemctl restart loc-pipeline.service

echo "=== 6. Status check ==="
systemctl status loc-pipeline.service --no-pager
echo ""
echo "=== Pipeline successfully launched! Tail logs with: tail -f /var/log/chronam-pipeline.log ==="
