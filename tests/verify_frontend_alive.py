import requests
import sys
import time

URL = "http://localhost:5173"

def check_frontend():
    print(f"--- CHECKING FRONTEND HEALTH: {URL} ---")
    try:
        response = requests.get(URL, timeout=2.0)
        
        # 1. Check HTTP 200
        if response.status_code != 200:
            print(f"  [FAILED] HTTP Status Code: {response.status_code}")
            sys.exit(1)
            
        print("  [PASSED] HTTP 200 OK")
        
        # 2. Check Content Size (Empty black screen body might be small, but Vite serves index.html)
        content = response.text
        if len(content) < 300:
            print(f"  [WARNING] Content size suspiciously small ({len(content)} bytes). Possible blank page.")
        else:
            print(f"  [PASSED] Content Size: {len(content)} bytes")
            
        # 3. Check for Title (in index.html)
        if "GLASC" in content or "Vite" in content:
             print("  [PASSED] Found expected title/meta tags.")
        else:
             print("  [WARNING] 'GLASC' or 'Vite' keywords not found in HTML.")
             
        print("\nNOTE: This script verifies the SERVER is running and serving HTML.")
        print("      To verify React rendering (avoiding Black Screen), use the 'ErrorBoundary' in the UI")
        print("      or run the specific 'tests/verify_defender_standalone.py' for backend logic.")
        print("--- FRONTEND ALIVE ---")
        
    except requests.exceptions.ConnectionError:
        print("  [FAILED] Connection Refused. Is 'npm run dev' running?")
        sys.exit(1)
    except Exception as e:
        print(f"  [FAILED] Unexpected Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_frontend()
