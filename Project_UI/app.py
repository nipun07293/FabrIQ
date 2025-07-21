from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
import os
import recommender
from werkzeug.utils import secure_filename
import uuid
import threading
import style_transfer

# ==============================================================================
# Flask App Initialization and Configuration
# ==============================================================================
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# --- Configuration ---
CSV_PATH = 'data.csv'
# IMPORTANT: Update this path to your actual folder of product images
IMAGE_FOLDER = r'D:/Python/Projects/AIMS_Summer/Project_2/UI/image'
MODEL_DATA_DIR = 'model_data'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global variable for product data ---
clothing_data = None
jobs = {}

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_data():
    """
    Load data from CSV, find all images in the image folder, and merge them.
    Images not in the CSV are given default "Fashion Outfit" metadata.
    """
    global clothing_data
    print("--- Starting Data Loading Process ---")
    
    try:
        # Step 1: Load the existing CSV data
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
        
        df_csv = pd.read_csv(CSV_PATH)
        # Standardize missing values immediately
        df_csv.replace(['Unknown', 'unknown', ''], pd.NA, inplace=True)
        df_csv.set_index('file_name', inplace=True)
        print(f"Loaded {len(df_csv)} records from {CSV_PATH}.")

        # Step 2: Scan the image directory for all image files
        if not os.path.isdir(IMAGE_FOLDER):
            raise FileNotFoundError(f"Image folder not found at {IMAGE_FOLDER}")
            
        all_image_files = {f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))}
        print(f"Found {len(all_image_files)} total images in {IMAGE_FOLDER}.")

        # Step 3: Find images that are in the folder but NOT in the CSV
        csv_image_files = set(df_csv.index)
        extra_image_files = all_image_files - csv_image_files
        print(f"Found {len(extra_image_files)} images not listed in the CSV.")

        # Step 4: Create a DataFrame for these extra images with default data
        if extra_image_files:
            extra_data = []
            for filename in extra_image_files:
                extra_data.append({
                    'file_name': filename,
                    'item': 'Fashion Outfit', # Default name
                    'clip_brand': 'Unknown Brand', # Default brand
                    # Add other columns as None (or pd.NA) so they exist
                    'category_name': 'Apparel',
                    'looks': None, 'colors': None, 'prints': None, 'sleeveLength': None,
                    'neckLine': None, 'fit': None, 'length': None, 'textures': None, 'shape': None
                })
            
            df_extra = pd.DataFrame(extra_data)
            df_extra.set_index('file_name', inplace=True)
            
            # Step 5: Combine the CSV data with the extra image data
            clothing_data = pd.concat([df_csv, df_extra])
        else:
            clothing_data = df_csv
        
        print(f"Successfully loaded a total of {len(clothing_data)} products.")

    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        clothing_data = pd.DataFrame()

def safe_to_dict(df):
    """
    Converts a DataFrame to a list of dictionaries, safely handling NaN values
    by converting them to None (which becomes JSON null).
    """
    # Convert all NaN/NaT to None before creating the dictionary list
    df_with_none = df.astype(object).where(pd.notnull(df), None)
    return df_with_none.to_dict('records')



def run_nst_in_background(job_id, content_path, style_path, result_path):
    """A wrapper function to run the NST process and update job status."""
    try:
        # Update status during the process
        def update_status_callback(message):
            if job_id in jobs:
                jobs[job_id]['status_message'] = message
        
        # Run the main NST function
        style_transfer.run_style_transfer(content_path, style_path, result_path, update_status_callback)
        
        # If successful, update the job status
        if job_id in jobs:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['status_message'] = 'Finished!'
            print(f"Job {job_id} completed successfully.")
            
    except Exception as e:
        # If it fails, update the status to 'failed'
        print(f"Job {job_id} failed: {e}")
        if job_id in jobs:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['status_message'] = f"An error occurred: {e}"
    finally:
        # Clean up the original uploaded files
        if os.path.exists(content_path):
            os.remove(content_path)
        if os.path.exists(style_path):
            os.remove(style_path)







def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('templates', 'index.html')

@app.route('/api/products')
def get_products():
    """API endpoint to get all products."""
    if clothing_data is None or clothing_data.empty:
        return jsonify({'success': False, 'error': 'Product data not loaded.', 'data': []}), 500
    
    products_list = safe_to_dict(clothing_data.reset_index())
    return jsonify({'success': True, 'data': products_list})






@app.route('/style-transfer')
def style_transfer_page():
    """Serves the new HTML page for the NST feature."""
    return send_from_directory('templates', 'style_transfer.html')

@app.route('/api/nst/start', methods=['POST'])
def start_nst_job():
    """Receives images, starts the NST background job, and returns a job ID."""
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'success': False, 'error': 'Both content and style images are required.'}), 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'success': False, 'error': 'Please select both files.'}), 400

    if allowed_file(content_file.filename) and allowed_file(style_file.filename):
        # Create unique filenames to avoid conflicts
        job_id = str(uuid.uuid4())
        content_ext = os.path.splitext(secure_filename(content_file.filename))[1]
        style_ext = os.path.splitext(secure_filename(style_file.filename))[1]
        
        content_filename = f"{job_id}_content{content_ext}"
        style_filename = f"{job_id}_style{style_ext}"
        result_filename = f"{job_id}_result.jpg"

        content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

        content_file.save(content_path)
        style_file.save(style_path)

        # Store job info
        jobs[job_id] = {
            'status': 'processing', 
            'status_message': 'Job started...',
            'result_filename': result_filename
        }
        
        # Start the background thread
        thread = threading.Thread(
            target=run_nst_in_background,
            args=(job_id, content_path, style_path, result_path)
        )
        thread.start()

        return jsonify({'success': True, 'job_id': job_id})
    else:
        return jsonify({'success': False, 'error': 'Invalid file type.'}), 400

@app.route('/api/nst/status/<job_id>')
def get_nst_job_status(job_id):
    """Provides the status of a background job to the frontend."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'success': False, 'error': 'Job ID not found.'}), 404
        
    response = {
        'success': True,
        'job_id': job_id,
        'status': job['status'],
        'status_message': job.get('status_message', ''),
        'result_url': f"/results/{job['result_filename']}" if job['status'] == 'completed' else None
    }
    return jsonify(response)

@app.route('/results/<path:filename>')
def serve_result_image(filename):
    """Serves the final stylized images from the 'results' folder."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)






@app.route('/api/filters')
def get_filter_options():
    """API endpoint to get all unique filter options."""
    if clothing_data is None or clothing_data.empty:
        return jsonify({'success': False, 'error': 'Product data not loaded.', 'filters': {}}), 500
    
    filters = {}
    filter_columns = {
        'category': 'category_name', 'item': 'item', 'brand': 'clip_brand',
        'looks': 'looks', 'colors': 'colors', 'material': 'prints',
        'sleeveLength': 'sleeveLength', 'neckLine': 'neckLine', 'fit': 'fit',
        'length': 'length', 'textures': 'textures', 'shape': 'shape'
    }
    
    data_for_filters = clothing_data.reset_index()
    for filter_key, column_name in filter_columns.items():
        if column_name in data_for_filters.columns:
            # Drop NA values before getting unique items to avoid 'None' as a filter option
            unique_values = data_for_filters[column_name].dropna().unique().tolist()
            filters[filter_key] = sorted([val for val in unique_values if val])
    
    return jsonify({'success': True, 'filters': filters})

@app.route('/api/recommend')
def recommend_products():
    """API endpoint for text or existing image recommendations."""
    query = request.args.get('query', '')
    query_type = request.args.get('type', 'text')

    if not query:
        return jsonify({'success': False, 'error': 'Query parameter is missing.'}), 400

    try:
        recommended_basenames = recommender.get_recommendations(query, input_type=query_type)
        if not recommended_basenames:
            return jsonify({'success': True, 'data': []})

        recs_df = clothing_data.loc[clothing_data.index.isin(recommended_basenames)].reindex(recommended_basenames)
        recs_list = safe_to_dict(recs_df.reset_index())
        
        return jsonify({'success': True, 'data': recs_list})
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500

@app.route('/api/recommend_by_upload', methods=['POST'])
def recommend_by_upload():
    """API endpoint for uploaded image recommendations."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in the request.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(upload_path)
            recommended_basenames = recommender.get_recommendations(upload_path, input_type='upload')
            
            if not recommended_basenames:
                return jsonify({'success': True, 'data': []})

            recs_df = clothing_data.loc[clothing_data.index.isin(recommended_basenames)].reindex(recommended_basenames)
            recs_list = safe_to_dict(recs_df.reset_index())
            
            return jsonify({'success': True, 'data': recs_list})
        except Exception as e:
            print(f"Error during upload recommendation: {e}")
            return jsonify({'success': False, 'error': 'An internal error occurred.'}), 500
        finally:
            if os.path.exists(upload_path):
                os.remove(upload_path)
            
    return jsonify({'success': False, 'error': 'File type not allowed.'}), 400

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve product images."""
    if not os.path.exists(IMAGE_FOLDER):
        return jsonify({'error': 'Image folder not found on server'}), 404
    return send_from_directory(IMAGE_FOLDER, filename)

# ==============================================================================
# Application Startup
# ==============================================================================
# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)



#     RESULTS_FOLDER = 'results'
#     app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     if not os.path.exists(RESULTS_FOLDER):
#         os.makedirs(RESULTS_FOLDER)


#     print("--- Server Starting ---")
#     load_data()
#     recommender.init_recommender(MODEL_DATA_DIR)
#     print(f"Serving images from: {os.path.abspath(IMAGE_FOLDER)}")
#     print(f"Reading data from: {os.path.abspath(CSV_PATH)}")
#     print("Access the UI at: http://127.0.0.1:5000")
#     print("-----------------------")
#     app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    # Create necessary folders on startup
    RESULTS_FOLDER = 'results'
    app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    
    print("--- Server Starting ---")
    
    # Load the CSV and image data, which is lightweight
    load_data()
    
    # REMOVED: recommender.init_recommender(MODEL_DATA_DIR)
    # The heavy models will now be loaded on the first API call instead of at startup.
    
    print(f"Serving images from: {os.path.abspath(IMAGE_FOLDER)}")
    print(f"Reading data from: {os.path.abspath(CSV_PATH)}")
    print("AI models will be loaded on first use.")
    print("Access the UI at: http://127.0.0.1:5000")
    print("-----------------------")
    
    # Run the app (keep use_reloader=False)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)