"""Static server for the deck that tells the browser never to cache, so a
plain refresh always picks up edited CSS and JS (Safari otherwise reuses its
cached copies)."""

import http.server
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8026
http.server.ThreadingHTTPServer(("", port), NoCache).serve_forever()
