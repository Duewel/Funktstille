from flask import Flask, request, send_file, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import os
import subprocess
import base64

UPLOAD_FOLDER = 'uploads'
MAPS_FOLDER = 'maps'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAPS_FOLDER'] = MAPS_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(MAPS_FOLDER):
    os.makedirs(MAPS_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result = subprocess.run(['python3', '0GesamtTesten.py'], capture_output=True, text=True)
        output_files = result.stdout.strip().split('\n')
        
        images_base64 = []
        for output_file in output_files:
            with open(output_file, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                images_base64.append(encoded_string)

        return jsonify({'plot': images_base64[0]})
    else:
        return jsonify({'error': 'File type not allowed'})

@app.route('/show_map', methods=['GET'])
def show_map():
    try:
        files = os.listdir(app.config['MAPS_FOLDER'])
        paths = [os.path.join(app.config['MAPS_FOLDER'], basename) for basename in files]
        latest_map_path = max(paths, key=os.path.getctime)
        map_url = url_for('static', filename=f'maps/{os.path.basename(latest_map_path)}')
        return jsonify({'success': True, 'map_url': map_url})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
