import os
from flask import Flask, send_from_directory, abort

app = Flask(__name__)

ROOT = os.path.dirname(__file__)


@app.route('/')
def portal():
    """Serve the hypoglycemia portal HTML file."""
    return send_from_directory(ROOT, 'hypoglycemia_portal.html')


@app.route('/<path:filename>')
def serve_file(filename):
    """Serve other files from repository root if they exist (assets, JS, CSS)."""
    path = os.path.join(ROOT, filename)
    if os.path.exists(path) and os.path.isfile(path):
        return send_from_directory(ROOT, filename)
    abort(404)


if __name__ == '__main__':
    # Use host 0.0.0.0 so it's reachable from other devices if needed
    app.run(host='0.0.0.0', port=5000, debug=True)
