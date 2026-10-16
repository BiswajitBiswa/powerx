#!/bin/zsh
# ------------------------------------------------------------------------------
# 1-Click Double-Clickable Joiner (ZERO PYTHON REQUIRED!)
# ------------------------------------------------------------------------------
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HEAD_IP="192.168.1.144"

# Native macOS Notification
osascript -e "display notification \"Connecting to Head Node at $HEAD_IP over Wi-Fi...\" with title \"⚡ Joining GPU Mesh\" sound name \"Glass\""

# Start the native zero-python worker in the background
nohup /bin/zsh "$DIR/worker.sh" "$HEAD_IP" 9401 >/dev/null 2>&1 &

# Friendly macOS Modal Alert
osascript -e "display alert \"Connected to GPU Mesh!\" message \"Your Mac is now safely connected to Node #01 ($HEAD_IP).\n\n• ZERO Python or developer tools required.\n• Automatically pauses in <0.3s when you touch your mouse.\n• NEVER runs while on battery power.\n\nYou can close this window.\""

# Auto-close the terminal window
osascript -e 'tell application "Terminal" to close (every window whose name contains "Join-Mesh")' &>/dev/null &
exit 0
