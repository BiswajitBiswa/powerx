#!/bin/bash
# -------------------------------------------------------------
# 1-Click Uninstaller for Local Nodes
# -------------------------------------------------------------
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="/usr/bin/python3"
fi

"$PYTHON_CMD" "$DIR/install_launchagent.py" uninstall

osascript -e 'display alert "PowerX Removed" message "PowerX has been completely uninstalled from this device."'
exit 0
