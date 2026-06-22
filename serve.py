import http.server, os
os.chdir(os.path.join(os.path.dirname(__file__), "output"))
http.server.test(HandlerClass=http.server.SimpleHTTPRequestHandler, port=8765, bind="127.0.0.1")
