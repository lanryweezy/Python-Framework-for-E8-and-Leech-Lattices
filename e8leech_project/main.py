import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from e8leech_project.api.lattice import lattice_bp

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development_only')
CORS(app)
app.register_blueprint(lattice_bp, url_prefix='/api/lattice')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_dir = app.static_folder
    if not static_dir:
        return "Static folder not configured", 404

    if path and os.path.exists(os.path.join(static_dir, path)):
       return send_from_directory(static_dir, path)

    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
       return send_from_directory(static_dir, 'index.html')

    return ("OK", 200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
