#!/bin/zsh
# ==============================================================================
# PowerX: Zero-Python Native macOS Worker Daemon
# ==============================================================================
# Runs on 100% of Macs out-of-the-box using native macOS tools (ioreg, pmset, sysctl, curl).
# Requires ZERO Python, ZERO Xcode, ZERO developer tools, and ZERO sudo!
# ==============================================================================

HEAD_IP="${1:-192.168.1.144}"
API_PORT="${2:-9401}"
HEAD_URL="http://${HEAD_IP}:${API_PORT}"

NODE_ID="mac-$(hostname -s | tr '[:upper:]' '[:lower:]')"
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon Mac")
TOTAL_MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "17179869184")
TOTAL_MEM_GB=$(( TOTAL_MEM_BYTES / 1073741824 ))

# Get GPU cores from ioreg
GPU_CORES=$(ioreg -r -c AGXAccelerator -d 2 2>/dev/null | grep -i '"gpu-core-count"' | awk -F'= ' '{print $2}' | tr -d ' ' || echo "10")
[ -z "$GPU_CORES" ] && GPU_CORES=10

echo "⚡ ============================================================="
echo "🍏 PowerX: Native macOS Worker (Zero-Python Mode)"
echo "⚡ ============================================================="
echo "  Node ID:       $NODE_ID"
echo "  Chip:          $CHIP_NAME"
echo "  GPU Cores:     $GPU_CORES"
echo "  Unified VRAM:  ${TOTAL_MEM_GB} GB"
echo "  Connecting To: $HEAD_URL"
echo "-------------------------------------------------------------"

# Background heartbeat & scavenging loop
while true; do
    # 1. Query user idle time in nanoseconds via IOHIDSystem
    IDLE_NS=$(ioreg -c IOHIDSystem 2>/dev/null | grep -i HIDIdleTime | head -n1 | awk '{print $NF}' || echo "0")
    IDLE_SEC=$(( IDLE_NS / 1000000000 ))

    # 2. Query AC power status via pmset
    BATT_INFO=$(pmset -g batt 2>/dev/null)
    IS_AC="false"
    if echo "$BATT_INFO" | grep -q "AC Power"; then
        IS_AC="true"
    fi

    # 3. Query GPU utilization from AGXAccelerator
    GPU_UTIL=$(ioreg -r -c AGXAccelerator -d 2 2>/dev/null | grep -i '"Device Utilization %"' | head -n1 | awk -F'= ' '{print $2}' | tr -d ' ' || echo "0")
    [ -z "$GPU_UTIL" ] && GPU_UTIL=0

    # 4. Determine if safe to scavenge: Idle > 60s, on AC power, GPU < 25%
    IS_IDLE="false"
    if [ "$IS_AC" = "true" ] && [ "$IDLE_SEC" -ge 60 ] && [ "$GPU_UTIL" -lt 25 ]; then
        IS_IDLE="true"
    fi

    # 5. Build JSON payload
    PAYLOAD=$(cat <<EOF
{
  "node_id": "$NODE_ID",
  "hardware_type": "apple_silicon",
  "device_name": "$CHIP_NAME",
  "gpu_cores": $GPU_CORES,
  "vram_total_gb": $TOTAL_MEM_GB,
  "vram_free_gb": $(( TOTAL_MEM_GB - 4 )),
  "gpu_util_pct": $GPU_UTIL,
  "user_idle_seconds": $IDLE_SEC,
  "is_idle": $IS_IDLE,
  "is_on_ac_power": $IS_AC
}
EOF
)

    # 6. Send heartbeat via curl to Head Node (silent, non-blocking)
    curl -s -X POST -H "Content-Type: application/json" \
         -d "$PAYLOAD" \
         --connect-timeout 2 \
         --max-time 3 \
         "${HEAD_URL}/api/heartbeat" >/dev/null 2>&1

    sleep 3
done
