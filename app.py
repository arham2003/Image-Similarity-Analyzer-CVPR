import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
from datetime import datetime

# Set GPU memory growth to avoid memory errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth enabled")

# Disable XLA optimization which might cause errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Flask app configuration
app = Flask(__name__)
app.secret_key = 'image_similarity_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global variable for the model
model = None

# Constants
IMG_SIZE = 224

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model_once():
    global model
    if model is None:
        try:
            model = keras.models.load_model('image_similarity_model', compile=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

# Calculate similarity percentage
def calculate_similarity_percentage(distance, scale=5.0):
    """Convert distance to a percentage similarity score"""
    similarity = np.exp(-distance / scale)
    return round(similarity * 100, 2)

def predict_model_similarity(img_path1, img_path2, threshold=0.7):
    """Predict similarity between two images using the model"""
    global model
    
    if not load_model_once():
        return None, None, None
        
    # Load and preprocess
    img1 = keras.preprocessing.image.load_img(img_path1, target_size=(IMG_SIZE, IMG_SIZE))
    img2 = keras.preprocessing.image.load_img(img_path2, target_size=(IMG_SIZE, IMG_SIZE))
    
    img1_array = keras.preprocessing.image.img_to_array(img1) / 255.0
    img2_array = keras.preprocessing.image.img_to_array(img2) / 255.0
    
    # Error handling for prediction
    try:
        # Predict embeddings
        emb1 = model.predict(np.expand_dims(img1_array, axis=0), verbose=0)
        emb2 = model.predict(np.expand_dims(img2_array, axis=0), verbose=0)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Using CPU for prediction")
        with tf.device('/CPU:0'):
            emb1 = model.predict(np.expand_dims(img1_array, axis=0), verbose=0)
            emb2 = model.predict(np.expand_dims(img2_array, axis=0), verbose=0)
    
    # Compute distance
    distance = np.linalg.norm(emb1 - emb2)
    similarity = calculate_similarity_percentage(distance)
    prediction = "Similar" if distance < threshold else "Not Similar"
    
    # Create visualization and save
    result_filename = f"{app.config['RESULTS_FOLDER']}/model_{uuid.uuid4().hex}.png"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis('off')
    
    plt.suptitle(f"Model: {prediction} ({similarity:.1f}% similarity, Distance: {distance:.2f})")
    plt.tight_layout()
    plt.savefig(result_filename)
    plt.close()
    
    return {
        'similarity': similarity,
        'prediction': prediction,
        'distance': distance,
        'visualization': os.path.basename(result_filename)
    }

def perform_sift_comparison(img_path1, img_path2, num_matches=50):
    """Performs SIFT feature extraction and matching"""
    # Load images
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)

    if image1 is None or image2 is None:
        return None
    
    # Generate unique filenames for results
    unique_id = uuid.uuid4().hex
    kp1_file = f"{app.config['RESULTS_FOLDER']}/kp1_{unique_id}.png"
    kp2_file = f"{app.config['RESULTS_FOLDER']}/kp2_{unique_id}.png"
    matches_file = f"{app.config['RESULTS_FOLDER']}/matches_{unique_id}.png"

    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and extract features
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Draw keypoints for image 1
    img_with_kp1 = cv2.drawKeypoints(
        image1, keypoints1, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_with_kp1 = cv2.cvtColor(img_with_kp1, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 4))
    plt.imshow(img_with_kp1)
    plt.title("Image 1 Keypoints")
    plt.axis('off')
    plt.savefig(kp1_file)
    plt.close()

    # Draw keypoints for image 2
    img_with_kp2 = cv2.drawKeypoints(
        image2, keypoints2, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_with_kp2 = cv2.cvtColor(img_with_kp2, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 4))
    plt.imshow(img_with_kp2)
    plt.title("Image 2 Keypoints")
    plt.axis('off')
    plt.savefig(kp2_file)
    plt.close()

    # Match descriptors
    if descriptors1 is None or descriptors2 is None:
        return None
        
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Limit matches to the available number
    num_matches = min(num_matches, len(matches))

    # Draw top matches
    matched_img = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches[:num_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    # Calculate match quality
    avg_distance = np.mean([m.distance for m in matches[:num_matches]]) if matches else float('inf')
    
    # Calculate SIFT similarity percentage (inverse of distance)
    max_sift_dist = 500  # Approximate max SIFT distance
    sift_similarity = max(0, 100 * (1 - avg_distance / max_sift_dist))
    
    # Save matches visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title(f"SIFT: {num_matches} matches, similarity: {sift_similarity:.1f}%, average distance: {avg_distance:.2f}")
    plt.axis('off')
    plt.savefig(matches_file)
    plt.close()
    
    return {
        'num_matches': num_matches,
        'avg_distance': avg_distance,
        'similarity': sift_similarity,
        'kp1_image': os.path.basename(kp1_file),
        'kp2_image': os.path.basename(kp2_file),
        'matches_image': os.path.basename(matches_file)
    }

@app.route('/', methods=['GET'])
def index():
    # Check if model is available
    model_available = load_model_once()
    return render_template('index.html', model_available=model_available)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file parts
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    # Check if user selected files
    if file1.filename == '' or file2.filename == '':
        flash('No selected files', 'danger')
        return redirect(request.url)
    
    # Check if both files are allowed
    if not (file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename)):
        flash('Invalid file type. Please upload png, jpg, or jpeg files.', 'danger')
        return redirect(request.url)
    
    # Get threshold value or use default
    threshold = float(request.form.get('threshold', 0.7))
    
    # Generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename1 = f"{timestamp}_1_{secure_filename(file1.filename)}"
    filename2 = f"{timestamp}_2_{secure_filename(file2.filename)}"
    
    filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    
    # Save the files
    file1.save(filepath1)
    file2.save(filepath2)
    
    # Store file information in the session for the results page
    session['filepath1'] = filepath1
    session['filepath2'] = filepath2
    session['filename1'] = filename1
    session['filename2'] = filename2
    session['threshold'] = threshold
    
    # Redirect to processing page
    return redirect(url_for('process_images'))

@app.route('/processing')
def process_images():
    # Check if files exist in session
    if 'filepath1' not in session or 'filepath2' not in session:
        flash('No images to process', 'danger')
        return redirect(url_for('index'))
    
    # Render the processing template - will auto-redirect to results
    return render_template('processing.html')

@app.route('/process', methods=['GET'])
def process_images_backend():
    # Check if files exist in session
    if 'filepath1' not in session or 'filepath2' not in session:
        flash('No images to process', 'danger')
        return redirect(url_for('index'))
    
    filepath1 = session['filepath1']
    filepath2 = session['filepath2']
    threshold = session['threshold']
    
    # Run the model prediction
    model_results = predict_model_similarity(filepath1, filepath2, threshold)
    
    # Run the SIFT comparison
    sift_results = perform_sift_comparison(filepath1, filepath2)
    
    # Convert any NumPy values to native Python types for JSON serialization
    if model_results:
        model_results['similarity'] = float(model_results['similarity'])
        model_results['distance'] = float(model_results['distance'])
    
    if sift_results:
        sift_results['num_matches'] = int(sift_results['num_matches'])
        sift_results['avg_distance'] = float(sift_results['avg_distance'])
        sift_results['similarity'] = float(sift_results['similarity'])
    
    # Store results in session
    session['model_results'] = model_results
    session['sift_results'] = sift_results
    
    # Redirect to results page
    return redirect(url_for('show_results'))

@app.route('/results')
def show_results():
    # Check if results exist in session
    if 'model_results' not in session or 'sift_results' not in session:
        flash('No results to display', 'danger')
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                           filename1=session['filename1'],
                           filename2=session['filename2'],
                           threshold=session['threshold'],
                           model_results=session['model_results'],
                           sift_results=session['sift_results'])

@app.route('/clear')
def clear_session():
    # Clear the session data
    session.clear()
    flash('Session cleared', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Load model at startup
    load_model_once()
    app.run(debug=True) 