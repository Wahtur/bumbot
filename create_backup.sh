#!/usr/bin/env bash
set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/bumbot/backups/bumbot_v1_${TIMESTAMP}"

mkdir -p "${BACKUP_DIR}/checkpoints"

echo "Backing up code and latest checkpoint to ${BACKUP_DIR}..."
cp /opt/bumbot/*.py "${BACKUP_DIR}/" 2>/dev/null || true
cp /opt/bumbot/*.sh "${BACKUP_DIR}/" 2>/dev/null || true
cp /opt/bumbot/training_log.csv "${BACKUP_DIR}/" 2>/dev/null || true
cp /opt/bumbot/checkpoints/weights_latest.pt "${BACKUP_DIR}/checkpoints/" 2>/dev/null || true
cp /opt/bumbot/checkpoints/weights_game_13400.pt "${BACKUP_DIR}/checkpoints/" 2>/dev/null || true

chmod -R 775 "${BACKUP_DIR}"
find "${BACKUP_DIR}" -type d -exec chmod 2775 {} +

echo "Backup complete!"
ls -la "${BACKUP_DIR}"
ls -la "${BACKUP_DIR}/checkpoints"
