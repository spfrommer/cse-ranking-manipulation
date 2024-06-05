from flask import Flask, send_from_directory, abort, render_template, make_response
from flask import request, jsonify
from flask_httpauth import HTTPBasicAuth

import os

pages_directory = './served_pages'
os.makedirs(pages_directory, exist_ok=True)


app = Flask(__name__)
auth = HTTPBasicAuth()


users = {
    'admin': '***ENTER PASSWORD***'
}


@auth.verify_password
def verify_password(username, password):
    if username in users and users[username] == password:
        return username

@app.route('/<path:filename>')
def serve_page(filename):
    if not filename.endswith('.html'):
        abort(404)  # Only serve HTML files

    safe_path = os.path.join(pages_directory, filename)
    if os.path.exists(safe_path) and os.path.isfile(safe_path):
        return send_from_directory(pages_directory, filename)
    else:
        abort(404)

@app.route('/upload/<filename>', methods=['PUT'])
@auth.login_required
def upload_file(filename):
    if not filename.endswith('.html'):
        abort(400, description='Only HTML files are allowed.')

    file_path = os.path.join(pages_directory, filename)
    with open(file_path, 'wb') as f:
        f.write(request.data)
    return jsonify(message='File uploaded successfully'), 201

@app.route('/clear', methods=['POST'])
@auth.login_required
def clear_files():
    for file_name in os.listdir(pages_directory):
        file_path = os.path.join(pages_directory, file_name)
        os.unlink(file_path)
    return jsonify(message='All files cleared'), 200

@app.route("/")
def home():
    response = make_response('Home', 200)
    response.mimetype = "text/plain"
    return response

if __name__ == '__main__':
    context = ('./letsencrypt/fullchain.pem', './letsencrypt/privkey.pem')
    # context = ('cert.pem', 'key.pem')
    app.run(host='0.0.0.0', port=443, debug=True, ssl_context=context)
    # app.run(host='0.0.0.0', port=80, debug=True)
