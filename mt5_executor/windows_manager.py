"""
Windows VPS Manager - Control ForexVPS from Linux via SSH

This script allows Claude Code to:
- Check Windows VPS status
- Start/stop MT5 Executor
- Monitor MT5 connection
- View logs
- Execute PowerShell commands

Usage:
    python windows_manager.py status
    python windows_manager.py start-executor
    python windows_manager.py stop-executor
    python windows_manager.py logs
    python windows_manager.py exec "Get-Process"

Environment Variables:
    WINDOWS_VPS_IP: ForexVPS IP address
    WINDOWS_VPS_USER: SSH username (default: Administrator)
    WINDOWS_VPS_KEY: Path to SSH private key (optional)
    WINDOWS_VPS_PASS: SSH password (if no key)
"""

import os
import sys
import subprocess
import argparse
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class WindowsVPSConfig:
    """Configuration for Windows VPS connection."""
    host: str = os.getenv("WINDOWS_VPS_IP", "")
    user: str = os.getenv("WINDOWS_VPS_USER", "Administrator")
    key_file: str = os.getenv("WINDOWS_VPS_KEY", "")
    password: str = os.getenv("WINDOWS_VPS_PASS", "")
    executor_port: int = 5000
    executor_path: str = r"C:\HYDRA\executor_service.py"


config = WindowsVPSConfig()


def ssh_command(cmd: str, use_powershell: bool = True) -> Tuple[int, str, str]:
    """
    Execute command on Windows VPS via SSH.

    Args:
        cmd: Command to execute
        use_powershell: Wrap in PowerShell (default True)

    Returns:
        (return_code, stdout, stderr)
    """
    if not config.host:
        return 1, "", "WINDOWS_VPS_IP not set"

    # Build SSH command
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]

    if config.key_file:
        ssh_args.extend(["-i", config.key_file])

    ssh_args.append(f"{config.user}@{config.host}")

    # Wrap command in PowerShell if needed
    if use_powershell:
        cmd = f'powershell -Command "{cmd}"'

    ssh_args.append(cmd)

    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "SSH connection timeout"
    except Exception as e:
        return 1, "", str(e)


def check_ssh_connection() -> bool:
    """Test SSH connection to Windows VPS."""
    code, out, err = ssh_command("echo 'connected'", use_powershell=False)
    return code == 0 and "connected" in out


def get_vps_status() -> dict:
    """Get comprehensive Windows VPS status."""
    status = {
        "ssh_connected": False,
        "mt5_running": False,
        "executor_running": False,
        "executor_api_healthy": False,
        "cpu_usage": None,
        "memory_usage": None,
        "uptime": None
    }

    # Check SSH
    if not check_ssh_connection():
        return status
    status["ssh_connected"] = True

    # Check if MT5 is running
    code, out, _ = ssh_command("Get-Process -Name terminal64 -ErrorAction SilentlyContinue | Select-Object -Property Name")
    status["mt5_running"] = "terminal64" in out

    # Check if Python executor is running
    code, out, _ = ssh_command("Get-Process -Name python -ErrorAction SilentlyContinue | Select-Object -Property Name")
    status["executor_running"] = "python" in out

    # Check executor API health
    code, out, _ = ssh_command(f"Invoke-WebRequest -Uri http://localhost:{config.executor_port}/health -UseBasicParsing -TimeoutSec 5 | Select-Object -ExpandProperty Content")
    status["executor_api_healthy"] = "ok" in out.lower()

    # Get system stats
    code, out, _ = ssh_command("Get-CimInstance Win32_Processor | Select-Object -ExpandProperty LoadPercentage")
    try:
        status["cpu_usage"] = int(out.strip())
    except (ValueError, AttributeError):
        pass

    code, out, _ = ssh_command("(Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize | ForEach-Object { [math]::Round((1 - ($_.FreePhysicalMemory / $_.TotalVisibleMemorySize)) * 100, 1) })")
    try:
        status["memory_usage"] = float(out.strip())
    except (ValueError, AttributeError):
        pass

    return status


def start_mt5_terminal() -> bool:
    """Start MT5 Terminal on Windows VPS."""
    # Find MT5 path
    code, out, _ = ssh_command("Get-ChildItem 'C:\\Program Files\\' -Filter terminal64.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName")

    if code != 0 or not out.strip():
        print("MT5 terminal not found. Looking in other locations...")
        code, out, _ = ssh_command("Get-ChildItem 'C:\\' -Filter terminal64.exe -Recurse -ErrorAction SilentlyContinue -Depth 3 | Select-Object -First 1 -ExpandProperty FullName")

    mt5_path = out.strip()
    if not mt5_path:
        print("ERROR: Could not find MT5 terminal")
        return False

    print(f"Found MT5 at: {mt5_path}")

    # Start MT5
    code, out, err = ssh_command(f"Start-Process '{mt5_path}'")
    if code == 0:
        print("MT5 Terminal started")
        return True
    else:
        print(f"Failed to start MT5: {err}")
        return False


def start_executor() -> bool:
    """Start the HYDRA MT5 Executor service."""
    # Check if already running
    code, out, _ = ssh_command("Get-Process -Name python -ErrorAction SilentlyContinue")
    if "python" in out:
        print("Executor already running")
        return True

    # Start executor in background
    cmd = f"""
    $env:FTMO_LOGIN = '{os.getenv("FTMO_LOGIN", "")}'
    $env:FTMO_PASS = '{os.getenv("FTMO_PASS", "")}'
    $env:FTMO_SERVER = '{os.getenv("FTMO_SERVER", "FTMO-Server3")}'
    $env:API_SECRET = '{os.getenv("MT5_API_SECRET", "hydra_secret_2024")}'
    Start-Process python -ArgumentList '{config.executor_path}' -WindowStyle Hidden
    """

    code, out, err = ssh_command(cmd)
    if code == 0:
        print("Executor started")
        return True
    else:
        print(f"Failed to start executor: {err}")
        return False


def stop_executor() -> bool:
    """Stop the HYDRA MT5 Executor service."""
    code, out, err = ssh_command("Stop-Process -Name python -Force -ErrorAction SilentlyContinue")
    print("Executor stopped" if code == 0 else f"Stop failed: {err}")
    return code == 0


def get_executor_logs(lines: int = 50) -> str:
    """Get recent executor logs."""
    # Try to get from Windows Event Log or console output
    code, out, _ = ssh_command(f"Get-Content 'C:\\HYDRA\\executor.log' -Tail {lines} -ErrorAction SilentlyContinue")
    if out.strip():
        return out
    return "No logs available (executor may not be logging to file)"


def execute_powershell(cmd: str) -> Tuple[int, str]:
    """Execute arbitrary PowerShell command."""
    code, out, err = ssh_command(cmd)
    return code, out if out else err


def print_status(status: dict):
    """Pretty print VPS status."""
    print("\n" + "=" * 50)
    print("     WINDOWS VPS STATUS (ForexVPS)")
    print("=" * 50)
    print(f"  SSH Connection:    {'✅ Connected' if status['ssh_connected'] else '❌ Disconnected'}")
    print(f"  MT5 Terminal:      {'✅ Running' if status['mt5_running'] else '❌ Not Running'}")
    print(f"  Executor Service:  {'✅ Running' if status['executor_running'] else '❌ Not Running'}")
    print(f"  Executor API:      {'✅ Healthy' if status['executor_api_healthy'] else '❌ Unhealthy'}")
    if status['cpu_usage'] is not None:
        print(f"  CPU Usage:         {status['cpu_usage']}%")
    if status['memory_usage'] is not None:
        print(f"  Memory Usage:      {status['memory_usage']}%")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Windows VPS Manager for HYDRA MT5")
    parser.add_argument("command", choices=[
        "status", "start-mt5", "start-executor", "stop-executor",
        "restart-executor", "logs", "exec", "test"
    ], help="Command to execute")
    parser.add_argument("args", nargs="*", help="Additional arguments")

    args = parser.parse_args()

    if not config.host:
        print("ERROR: WINDOWS_VPS_IP environment variable not set")
        print("Set it with: export WINDOWS_VPS_IP=your_forexvps_ip")
        sys.exit(1)

    if args.command == "status":
        status = get_vps_status()
        print_status(status)

    elif args.command == "start-mt5":
        start_mt5_terminal()

    elif args.command == "start-executor":
        start_executor()

    elif args.command == "stop-executor":
        stop_executor()

    elif args.command == "restart-executor":
        stop_executor()
        import time
        time.sleep(2)
        start_executor()

    elif args.command == "logs":
        lines = int(args.args[0]) if args.args else 50
        print(get_executor_logs(lines))

    elif args.command == "exec":
        if not args.args:
            print("Usage: windows_manager.py exec 'PowerShell command'")
            sys.exit(1)
        cmd = " ".join(args.args)
        code, output = execute_powershell(cmd)
        print(output)
        sys.exit(code)

    elif args.command == "test":
        print(f"Testing SSH connection to {config.user}@{config.host}...")
        if check_ssh_connection():
            print("✅ SSH connection successful!")
        else:
            print("❌ SSH connection failed")
            print("\nTroubleshooting:")
            print("1. Ensure OpenSSH Server is installed on Windows")
            print("2. Check firewall allows port 22")
            print("3. Verify credentials")
            sys.exit(1)


if __name__ == "__main__":
    main()
