#!/usr/bin/env bash
set -euxo pipefail

APP_DIR="/home/ubuntu/loc-chronicling-america"
cd "${APP_DIR}"
git config --global --add safe.directory "${APP_DIR}" || true
git pull origin main

"${APP_DIR}/.venv/bin/python" -c "from loc_chronicling_america.db import CatalogDB; db = CatalogDB(); n = db.mark_failed_batches_pending(); print(f'Reset {n} failed batches to pending')"

systemctl restart loc-pipeline
echo "=== Worker updated and pipeline restarted successfully ==="
