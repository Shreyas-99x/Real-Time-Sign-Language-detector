import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8000
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

class SignSpeakHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        # Enable CORS and disable cache during development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def main():
    # Use 127.0.0.1 explicitly to avoid IPv6 localhost resolution mismatch on Windows
    server_address = ("127.0.0.1", PORT)
    try:
        httpd = HTTPServer(server_address, SignSpeakHandler)
    except OSError as e:
        if e.winerror == 10048 or 'already in use' in str(e).lower():
            # Port 8000 is in use, try 8080
            alt_port = 8080
            print(f"Port {PORT} is busy, switching to port {alt_port}...")
            server_address = ("127.0.0.1", alt_port)
            httpd = HTTPServer(server_address, SignSpeakHandler)
        else:
            raise e

    actual_port = httpd.server_port
    url = f"http://127.0.0.1:{actual_port}"
    print(f"\n=======================================================")
    print(f"  SignSpeak is running at: {url}")
    print(f"  Serving directory: {WEB_DIR}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")

    # Automatically open in the user's default browser
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SignSpeak server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
