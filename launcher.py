#!/usr/bin/env python3
"""Simple CLI launcher for the RL project."""

import sys
import os
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🚀 {description}")
    print("-" * 50)
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False

def main():
    """Main CLI launcher."""
    parser = argparse.ArgumentParser(description="RL Project Launcher")
    parser.add_argument("command", choices=[
        "install", "train", "streamlit", "notebook", "test", "example", "help"
    ], help="Command to run")
    parser.add_argument("--env", type=str, default="FrozenLake-v1", 
                      help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, 
                      help="Number of episodes")
    parser.add_argument("--timesteps", type=int, default=10000, 
                      help="Number of timesteps")
    
    args = parser.parse_args()
    
    print("🤖 Reinforcement Learning Project Launcher")
    print("=" * 50)
    
    if args.command == "install":
        print("📦 Installing dependencies...")
        run_command("pip install -r requirements.txt", "Dependency installation")
        
    elif args.command == "train":
        print("🏋️ Training agents...")
        cmd = f"python src/main.py --env {args.env} --episodes {args.episodes}"
        run_command(cmd, "Agent training")
        
    elif args.command == "streamlit":
        print("🌐 Launching Streamlit interface...")
        run_command("streamlit run src/streamlit_app.py", "Streamlit interface")
        
    elif args.command == "notebook":
        print("📓 Launching Jupyter notebook...")
        run_command("jupyter notebook notebooks/", "Jupyter notebook")
        
    elif args.command == "test":
        print("🧪 Running tests...")
        run_command("python -m pytest tests/ -v", "Unit tests")
        
    elif args.command == "example":
        print("📚 Running comprehensive example...")
        cmd = f"python src/comprehensive_example.py --env {args.env} --episodes {args.episodes} --timesteps {args.timesteps}"
        run_command(cmd, "Comprehensive example")
        
    elif args.command == "help":
        print("""
📖 Available Commands:

  install     - Install project dependencies
  train       - Train SARSA and Q-Learning agents
  streamlit   - Launch interactive web interface
  notebook    - Launch Jupyter notebook for analysis
  test        - Run unit tests
  example     - Run comprehensive comparison example
  help        - Show this help message

📋 Usage Examples:

  python launcher.py install
  python launcher.py train --env FrozenLake-v1 --episodes 1000
  python launcher.py streamlit
  python launcher.py example --env CartPole-v1 --timesteps 5000

🌍 Supported Environments:
  - FrozenLake-v1 (tabular methods)
  - CartPole-v1 (advanced methods)
  - MountainCar-v0 (advanced methods)
  - GridWorld (custom tabular)
  - CliffWalking (custom tabular)
        """)

if __name__ == "__main__":
    main()
