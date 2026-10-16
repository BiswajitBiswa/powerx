#!/bin/bash
# -------------------------------------------------------------
# 1-Click Installer for Local Nodes
# -------------------------------------------------------------
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$(dirname "$DIR")"

echo "⚡ Setting up PowerX background service..."

# Use system python3 or virtualenv python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    osascript -e 'display alert "Python 3 Not Found" message "Please install Python 3 or contact IT support."'
    exit 1
fi

"$PYTHON_CMD" "$DIR/install_launchagent.py"

# Native user notification popup
osascript -e 'display notification "PowerX is now active in the background. It will automatically pause whenever you use this device." with title "⚡ PowerX Active" sound name "Glass"'

# Small friendly alert modal
osascript -e 'display alert "PowerX Installed!" message "This node is now contributing idle compute to the PowerX cluster.\n\n• It automatically pauses in <0.3s when user activity is detected.\n• It NEVER runs while on battery power.\n• It starts silently in the background on every login."'

echo "Done! You can close this window."
osascript -e 'tell application "Terminal" to close (every window whose name contains "Install-GPU-Mesh")' &>/dev/null &
exit 0
