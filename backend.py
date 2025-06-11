from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
import sqlite3
import os
from io import BytesIO
import tempfile
import uuid
import re

app = Flask(__name__)

# This would be set from the frontend
GEMINI_API_KEY = None
UPLOADED_FILES = {}

def _generate_id():
    return str(uuid.uuid4())

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    global GEMINI_API_KEY
    data = request.json
    api_key = data.get('api_key')
    if api_key:
        GEMINI_API_KEY = api_key
        return jsonify({"message": "API key set successfully"}), 200
    return jsonify({"error": "No API key provided"}), 400

@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file
    file = request.files['file']
    
    # Process different file types
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    elif file.filename.endswith('.db'):
        # For SQLite, we'll connect and get the first table as example
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            conn = sqlite3.connect(tmp.name)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                # Get the first table
                table_name = tables[0][0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
                return jsonify({"error": "No tables found in SQLite database"}), 400
            conn.close()
            os.unlink(tmp.name)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    
    # Store the dataframe with a unique ID
    file_id = _generate_id()
    UPLOADED_FILES[file_id] = df
    
    # Sample the data and convert to dict for JSON response
    sample_data = df.head().to_dict(orient='records')
    
    return jsonify({
        "message": "File processed successfully",
        "file_id": file_id,
        "columns": list(df.columns),
        "sample_data": sample_data,
        "row_count": len(df),
        "data_quality": {
            "completeness": check_completeness(df),
            "uniqueness": check_uniqueness(df)
        }
    })

def check_completeness(df):
    # Simple data quality check - % of non-null values
    return (1 - df.isnull().mean()).to_dict()

def check_uniqueness(df):
    # Check uniqueness of values in each column
    return (df.nunique() / len(df)).to_dict()

@app.route('/analyze', methods=['POST'])
def analyze_data():
    data = request.json
    query = data.get('query')
    file_id = data.get('file_id')
    
    if not file_id or file_id not in UPLOADED_FILES:
        return jsonify({"error": "No file uploaded or invalid file ID"}), 400
        
    df = UPLOADED_FILES[file_id] # Retrieve the dataframe
    
    if not GEMINI_API_KEY:
        return jsonify({"error": "Gemini API key not configured"}), 400
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    
    # Convert the dataframe to a string representation for the prompt
    df_str = df.to_string() # Convert entire dataframe to string
    
    prompt = f"""You are a data analysis assistant. Analyze this dataset and answer the user's query.
    
Dataset sample:
{df_str}

User query: {query}

Respond with:
1. A clear answer to the query
2. Recommended visualization types (choose from: bar, line, pie, scatter, heatmap)
3. 3 key insights from the data
4. Data quality considerations
"""
    
    try:
        response = model.generate_content(prompt)
        
        # Parse Gemini's response
        response_text = response.text
        
        # Initialize default values
        answer = "No answer generated."
        charts = []
        insights = []
        quality = 0.5 # Default quality score
        
        # Attempt to parse structured information from the response
        lines = response_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('1. '):
                answer = line[3:].strip()
                current_section = 'answer'
            elif line.startswith('2. '):
                charts_str = line[3:].strip()
                charts = [chart.strip() for chart in charts_str.replace('Recommended visualization types (choose from: bar, line, pie, scatter, heatmap):', '').split(',') if chart.strip()]
                current_section = 'charts'
            elif line.startswith('3. '):
                current_section = 'insights'
            elif line.startswith('4. '):
                current_section = 'quality'
            elif current_section == 'insights' and line and not line.startswith(tuple([str(i) + '.' for i in range(1, 10)])): # Check if it's not a new numbered section
                insights.append(line.strip('-').strip())
            elif current_section == 'quality' and line and not line.startswith(tuple([str(i) + '.' for i in range(1, 10)])):
                try:
                    if "confidence score" in line.lower():
                        quality = float(line.split(':')[-1].replace('%', '').strip()) / 100
                    elif "quality" in line.lower():
                        # Look for numerical values in the line
                        match = re.search(r'\d+\.?\d*%', line)
                        if match:
                            quality = float(match.group().replace('%', '')) / 100
                        else:
                            match = re.search(r'\d+\.?\d*', line)
                            if match:
                                quality = float(match.group()) / 100 if float(match.group()) > 1 else float(match.group())
                except ValueError:
                    pass
        
        return jsonify({
            "answer": answer,
            "quality": quality,
            "charts": charts,
            "tables": df.head().to_dict(orient='records'), # Return sample data
            "insights": insights,
            "status": "success"
        })    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
