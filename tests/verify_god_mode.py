import requests
import websocket
import json
import threading
import time
import sys

# Configuration
API_URL = "http://localhost:8000/api/sim/inject_message"
WS_URL = "ws://localhost:8000/ws/monitor"
TEST_MESSAGE = "God Mode Test Sequence Initiated"

def on_message(ws, message):
    data = json.loads(message)
    if data.get("type") == "NEGOTIATION_EVENT":
        print(f"\n[WS] Received Event: {data['sender']} -> {data['content']}")
        if data['content'] == TEST_MESSAGE and data['sender'] == "USER_GOD_MODE":
            print("SUCCESS: God Mode message received via WebSocket!")
            ws.close()
            sys.exit(0)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket connection opened. Sending HTTP Injection...")
    def run():
        time.sleep(1) # Wait for connection to stabilize
        try:
            response = requests.post(API_URL, json={"content": TEST_MESSAGE, "target": "ALL"})
            print(f"HTTP POST Response: {response.status_code} {response.json()}")
        except Exception as e:
            print(f"HTTP POST Failed: {e}")
            ws.close()
            sys.exit(1)
    threading.Thread(target=run).start()

if __name__ == "__main__":
    # Check if server is up first
    try:
        requests.get("http://localhost:8000/docs", timeout=2)
    except:
        print("Server not running. Please start the backend API first.")
        sys.exit(1)

    print("Starting WebSocket Listener...")
    ws = websocket.WebSocketApp(WS_URL,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    
    # Run WS with timeout (if we don't get the message in 5s, fail)
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    
    time.sleep(5)
    if ws.keep_running:
        print("TIMEOUT: Dig not receive the God Mode message within 5 seconds.")
        ws.close()
        sys.exit(1)
