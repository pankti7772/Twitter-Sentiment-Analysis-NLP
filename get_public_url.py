#!/usr/bin/env python3
"""Get the public ngrok URL"""
import urllib.request
import json
import time
import sys

# Wait a bit for ngrok to start
time.sleep(2)

try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as response:
        data = json.loads(response.read().decode())
        if data.get("tunnels"):
            public_url = data["tunnels"][0]["public_url"]
            print(f"🌐 Public URL: {public_url}")
            sys.exit(0)
        else:
            print("No active tunnels found")
except urllib.error.URLError:
    print("⚠️  Ngrok web interface not accessible. The server may still be starting...")
    print("💡 Check the console output from 'python app.py' for the public URL")
    print("💡 Or make sure ngrok is running and authenticated")
except Exception as e:
    print(f"Error: {e}")

