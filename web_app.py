import os
import json
from flask import Flask, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import pandas as pd

from data_agent import DataAgent

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store DataAgent instances per session
agents = {}

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_or_create_agent(session_id):
    """Get existing DataAgent or create new one for session"""
    if session_id not in agents:
        try:
            agents[session_id] = DataAgent()
        except ValueError as e:
            return None, str(e)
    return agents[session_id], None

@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please upload CSV, Excel, JSON, or Parquet files.'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Get or create DataAgent
        agent, error = get_or_create_agent(session_id)
        if not agent:
            return jsonify({'error': f'DataAgent initialization failed: {error}'}), 500
        
        # Load data using DataAgent
        result = agent.execute_tool("load_data", {
            "filepath": filepath,
            "name": filename.rsplit('.', 1)[0]
        })
        
        if "Error" in result or "not found" in result:
            return jsonify({'error': result}), 400
        
        # Get data preview
        preview_result = agent.execute_tool("preview_data", {
            "name": filename.rsplit('.', 1)[0],
            "rows": 10
        })
        
        # Get data info
        info_result = agent.execute_tool("get_data_info", {
            "name": filename.rsplit('.', 1)[0]
        })
        
        return jsonify({
            'success': True,
            'filename': filename,
            'load_result': result,
            'preview': preview_result,
            'info': info_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if 'session_id' not in session:
        return jsonify({'error': 'Session not found'}), 400
    
    session_id = session['session_id']
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    message = data['message']
    
    try:
        # Get DataAgent for this session
        agent, error = get_or_create_agent(session_id)
        if not agent:
            return jsonify({'error': f'DataAgent not available: {error}'}), 500
        
        # Send message to DataAgent
        response = agent.send_message(message)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({'error': f'Chat failed: {str(e)}'}), 500

@app.route('/datasets')
def get_datasets():
    if 'session_id' not in session:
        return jsonify({'error': 'Session not found'}), 400
    
    session_id = session['session_id']
    agent, error = get_or_create_agent(session_id)
    
    if not agent:
        return jsonify({'datasets': []})
    
    # Get list of loaded datasets
    datasets = []
    for name, df in agent.dataframes.items():
        datasets.append({
            'name': name,
            'shape': df.shape,
            'columns': list(df.columns)
        })
    
    return jsonify({'datasets': datasets})

@app.route('/export/<dataset_name>')
def export_dataset(dataset_name):
    if 'session_id' not in session:
        return jsonify({'error': 'Session not found'}), 400
    
    session_id = session['session_id']
    agent, error = get_or_create_agent(session_id)
    
    if not agent or dataset_name not in agent.dataframes:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        # Save dataset to temporary file
        export_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{dataset_name}_export.csv")
        agent.dataframes[dataset_name].to_csv(export_path, index=False)
        
        return send_file(export_path, as_attachment=True, download_name=f"{dataset_name}_processed.csv")
        
    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    if 'session_id' not in session:
        return jsonify({'success': True})
    
    session_id = session['session_id']
    
    # Remove agent from memory
    if session_id in agents:
        del agents[session_id]
    
    # Clean up uploaded files for this session
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(session_id):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception:
        pass  # Ignore cleanup errors
    
    # Clear session
    session.clear()
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)