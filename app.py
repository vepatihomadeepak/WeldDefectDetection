#!/usr/bin/env python3
"""
Main entry point for AI Weld Seam Inspector.

Run with: python app.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch Streamlit app."""
    app_path = Path(__file__).parent / "src" / "app.py"
    
    if not app_path.exists():
        print(f"Error: App not found at {app_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("AI Weld Seam Inspector")
    print("=" * 60)
    print("Starting Streamlit application...")
    print("Open your browser at http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        completed = subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--browser.serverAddress", "localhost",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
        if completed.returncode:
            print(f"Streamlit exited with code {completed.returncode}.")
            sys.exit(completed.returncode)
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
