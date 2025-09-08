#!/usr/bin/env python3
"""
Launch script for DataAgent Web UI
"""

import os
import sys
from web_app import app

def main():
    print("🚀 Starting DataAgent Web UI...")
    print("📊 Upload CSV, Excel, JSON, or Parquet files")
    print("💬 Chat with your data using natural language")
    print("🌐 Open http://localhost:8000 in your browser")
    print("")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: No .env file found!")
        print("   Create a .env file with your ANTHROPIC_API_KEY")
        print("   Example:")
        print("   ANTHROPIC_API_KEY=your-key-here")
        print("")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        print("\n👋 DataAgent Web UI stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()