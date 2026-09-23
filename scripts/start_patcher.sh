#!/usr/bin/env bash
set -euxo pipefail

APP_DIR="/home/ubuntu/loc-chronicling-america"
cd "${APP_DIR}"
git config --global --add safe.directory "${APP_DIR}" || true
git pull origin main

# Source environment
if [ -f "${APP_DIR}/.env" ]; then
    set -a
    source "${APP_DIR}/.env"
    set +a
fi

LOG_FILE="/var/log/chronam-patcher.log"
touch "${LOG_FILE}"
chown ubuntu:ubuntu "${LOG_FILE}"

echo "=== Starting Parquet URL Patcher for all states ==="
echo "HF_REPO: ${HF_REPO:-Tim-Pinecone/LOC-Chronicling-America}"
echo "HF_TOKEN length: ${#HF_TOKEN}"

# Kill any existing patcher instance first
pkill -f "patch_parquet_urls.py" || true
sleep 1

# Launch detached in background with setsid so it survives SSM session end
setsid nice -n 15 "${APP_DIR}/.venv/bin/python" -u "${APP_DIR}/scripts/patch_parquet_urls.py" \
    --all-states \
    --workers 4 \
    --batch-size 500 \
    > "${LOG_FILE}" 2>&1 < /dev/null &

sleep 2
echo "=== Patcher process status ==="
pgrep -f "patch_parquet_urls.py" || true
echo "=== Initial Log Output (${LOG_FILE}) ==="
cat "${LOG_FILE}"
