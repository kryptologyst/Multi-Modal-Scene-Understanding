#!/usr/bin/env python3
"""Setup script for multi-modal scene understanding project."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of what the command does.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Multi-Modal Scene Understanding - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return 1
    
    print(f"✓ Python version: {sys.version}")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Setup failed at: {description}")
            return 1
    
    # Run setup test
    print("\nRunning setup test...")
    if not run_command("python test_setup.py", "Setup verification"):
        print("❌ Setup test failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test the demo: python demo/app.py")
    print("2. Run training: python scripts/train.py")
    print("3. Run evaluation: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
