"""Configures and registers PowerX as a native macOS LaunchAgent."""

import os
import subprocess
import sys
from pathlib import Path

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>io.powerx.mesh</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>-m</string>
        <string>hybrid_gpu_mesh.cli</string>
        <string>daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{repo_root}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ProcessType</key>
    <string>Background</string>
    <key>LowPriorityIO</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/gpu-mesh.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/gpu-mesh.err</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{repo_root}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{path_extra}</string>
    </dict>
</dict>
</plist>
"""

def install_launchagent():
    home = Path.home()
    launch_agents_dir = home / "Library" / "LaunchAgents"
    log_dir = home / "Library" / "Logs"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_path = launch_agents_dir / "io.powerx.mesh.plist"
    repo_root = Path(__file__).resolve().parent.parent
    python_bin = sys.executable

    # Unload previous instance if running
    if plist_path.exists():
        try:
            subprocess.run(["launchctl", "unload", str(plist_path)], stderr=subprocess.DEVNULL)
        except Exception:
            pass

    content = PLIST_TEMPLATE.format(
        python_bin=python_bin,
        repo_root=str(repo_root),
        log_dir=str(log_dir),
        path_extra=str(Path(python_bin).parent),
    )

    with open(plist_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Register and start with macOS launchd
    subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True)
    print(f"✅ Successfully installed and activated LaunchAgent: {plist_path}")
    print(f"📄 Logs are writing to: {log_dir / 'gpu-mesh.log'}")

def uninstall_launchagent():
    home = Path.home()
    plist_path = home / "Library" / "LaunchAgents" / "io.powerx.mesh.plist"
    if plist_path.exists():
        try:
            subprocess.run(["launchctl", "unload", "-w", str(plist_path)], check=True)
        except Exception:
            pass
        plist_path.unlink(missing_ok=True)
        print("🗑️ Uninstalled PowerX LaunchAgent.")
    else:
        print("No installed LaunchAgent found.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_launchagent()
    else:
        install_launchagent()
