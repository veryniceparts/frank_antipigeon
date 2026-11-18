#!/usr/bin/env bash
# GigE Vision host tuning for a Teledyne DALSA Genie Nano on Jetson (see docs/DALSA.md).
#   sudo bash scripts/gige_tune.sh eth0 [host_ip/prefix]     e.g.  sudo bash scripts/gige_tune.sh eth0 192.168.10.1/24
# Settings from 'ip'/'sysctl' are not persistent -> the script also writes /etc/sysctl.d and prints a netplan/nmcli hint.
set -euo pipefail
IF=${1:?usage: gige_tune.sh <interface> [host_ip/prefix]}
ADDR=${2:-}

if [ -n "$ADDR" ]; then
  ip addr flush dev "$IF"
  ip addr add "$ADDR" dev "$IF"
fi
ip link set "$IF" up

# jumbo frames: ~6x fewer packets per frame -> much less CPU per frame
if ip link set "$IF" mtu 9000 2>/dev/null; then
  echo "MTU 9000 on $IF"
else
  echo "WARNING: $IF refused MTU 9000; staying at $(cat /sys/class/net/$IF/mtu). Aravis will negotiate a smaller packet size."
fi

# large socket receive buffers (packet loss under load otherwise)
cat > /etc/sysctl.d/60-gige-vision.conf <<'EOF'
net.core.rmem_max=33554432
net.core.rmem_default=33554432
EOF
sysctl -q -p /etc/sysctl.d/60-gige-vision.conf

# let Aravis use packet sockets (lower CPU) without running as root
PY=$(readlink -f "$(command -v python3)")
if command -v setcap >/dev/null; then
  setcap cap_net_raw+ep "$PY" && echo "cap_net_raw set on $PY"
fi

# interrupt coalescing: fewer interrupts per frame (ignored if the driver doesn't support it)
ethtool -C "$IF" rx-usecs 50 2>/dev/null || true

echo
echo "Done. To persist IP/MTU, e.g. with NetworkManager:"
echo "  nmcli con mod <connection> 802-3-ethernet.mtu 9000 ipv4.method manual ipv4.addresses ${ADDR:-192.168.10.1/24}"
echo "Check the camera: arv-tool-0.8   (then: python3 scripts/run.py --source 'dalsa://<ip>?pf=BayerRG8&w=1280&h=480&exp=800&fps=300')"
