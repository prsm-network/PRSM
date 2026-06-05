#!/bin/bash
#
# PRSM Operator Node — cloud-init userdata (sprint 458)
#
# Spins up a PRSM operator node (NOT a bootstrap server).
# The operator node connects to the canonical bootstrap fleet,
# discovers peers, optionally publishes/retrieves content, and
# serves the operator REST + MCP surface on port 8000.
#
# Paste this into the cloud provider's "User data" field
# during instance launch. No SSH-in required for first boot.
#
# CUSTOMIZE BEFORE PASTING:
#   - NODE_DISPLAY_NAME (optional, defaults to "prsm-op-<random>")
#   - BOOTSTRAP_URL (defaults to canonical wss://bootstrap-us.prsm-network.com:8765)
#
# WATCH PROGRESS:
#   ssh -i <your_key> ubuntu@<PUBLIC_IP>
#   sudo tail -f /var/log/cloud-init-output.log
#
# VERIFY AFTER BOOT:
#   curl http://<PUBLIC_IP>:8000/info
#   curl http://<PUBLIC_IP>:8000/bootstrap/status
#   curl http://<PUBLIC_IP>:8000/peers
#
# Sprint 458 — created during the verification campaign's multi-host
# test bench work to close F14 (NAT-loopback blocked single-host
# direct P2P; cloud VM with public IP solves it).

set -euxo pipefail

# ────── CUSTOMIZE (optional) ──────
NODE_DISPLAY_NAME="prsm-op-cloud"
BOOTSTRAP_URL="wss://bootstrap-us.prsm-network.com:8765"
# ──────────────────────────────────

PRSM_REPO="https://github.com/prsm-network/PRSM.git"
INSTALL_DIR="/opt/prsm"
DATA_DIR="/var/lib/prsm"

# ── 1. System prep ──────────────────────────────────────
timedatectl set-timezone UTC
hostnamectl set-hostname "${NODE_DISPLAY_NAME}"

export DEBIAN_FRONTEND=noninteractive

# Wait for unattended-upgrades to release apt lock
for i in {1..30}; do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

apt-get update -qq
apt-get install -y -qq \
    python3 python3-venv python3-pip \
    git ufw \
    python3-libtorrent

# ── 2. Firewall ─────────────────────────────────────────
# 22/tcp  — SSH
# 8000/tcp — operator REST/MCP API
# 9001/tcp — P2P WebSocket (libp2p)
ufw allow 22/tcp
ufw allow 8000/tcp
ufw allow 9001/tcp
ufw --force enable

# ── 3. PRSM install ─────────────────────────────────────
mkdir -p "$INSTALL_DIR" "$DATA_DIR"
chown ubuntu:ubuntu "$INSTALL_DIR" "$DATA_DIR"

sudo -u ubuntu git clone --depth 1 "$PRSM_REPO" "$INSTALL_DIR"
sudo -u ubuntu python3 -m venv --system-site-packages "$INSTALL_DIR/.venv"
sudo -u ubuntu "$INSTALL_DIR/.venv/bin/pip" install --upgrade pip --quiet
sudo -u ubuntu "$INSTALL_DIR/.venv/bin/pip" install -e "$INSTALL_DIR" --quiet
sudo -u ubuntu "$INSTALL_DIR/.venv/bin/pip" install bencodepy --quiet

# ── 4. systemd unit ─────────────────────────────────────
cat > /etc/systemd/system/prsm-operator.service <<EOF
[Unit]
Description=PRSM Operator Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=${INSTALL_DIR}
Environment=HOME=/home/ubuntu
Environment=PRSM_DATA_DIR=${DATA_DIR}
Environment=PRSM_QUERY_ORCHESTRATOR_ENABLED=1
ExecStart=${INSTALL_DIR}/.venv/bin/python -m prsm.cli node start \\
    --no-dashboard \\
    --api-port 8000 \\
    --p2p-port 9001 \\
    --bootstrap ${BOOTSTRAP_URL}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=true
ProtectSystem=full
ProtectHome=read-only
PrivateTmp=true
ReadWritePaths=${DATA_DIR} /home/ubuntu/.prsm

[Install]
WantedBy=multi-user.target
EOF

# ── 5. Pre-seed data dir + start ────────────────────────
# Pre-create the home/.prsm path so the systemd unit's
# ReadWritePaths covers everything the daemon writes.
mkdir -p /home/ubuntu/.prsm
chown -R ubuntu:ubuntu /home/ubuntu/.prsm

systemctl daemon-reload
systemctl enable prsm-operator.service
systemctl start prsm-operator.service

# ── 6. Final status ─────────────────────────────────────
sleep 15  # let the daemon do its first boot

systemctl status prsm-operator.service --no-pager || true
echo ""
echo "════════════════════════════════════════════════════"
echo "PRSM Operator Node — first boot complete"
echo "════════════════════════════════════════════════════"
PUBLIC_IP=$(curl -s --max-time 5 http://checkip.amazonaws.com || echo "unknown")
echo ""
echo "Public IP: ${PUBLIC_IP}"
echo "API:       http://${PUBLIC_IP}:8000/info"
echo "Bootstrap: http://${PUBLIC_IP}:8000/bootstrap/status"
echo "Peers:     http://${PUBLIC_IP}:8000/peers"
echo ""
echo "Verify locally:"
echo "  curl http://${PUBLIC_IP}:8000/info"
echo ""
echo "Service logs:"
echo "  sudo journalctl -u prsm-operator -f"
echo "════════════════════════════════════════════════════"
