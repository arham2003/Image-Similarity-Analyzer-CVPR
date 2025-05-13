import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2
import argparse

# Set GPU memory growth to avoid memory errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth enabled")

# Disable XLA optimization which might cause errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Constants
IMG_SIZE = 224

# Load the saved model
def load_model(model_path='image_similarity_model'):
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    return model

# Calculate similarity percentage
def calculate_similarity_percentage(distance, scale=5.0):
    """Convert distance to a percentage similarity score"""
    similarity = np.exp(-distance / scale)
    return round(similarity * 100, 2)

# Predict similarity between two images using the model
def predict_similarity(model, image_path1, image_path2, threshold=0.7, show_images=True):
    """Predict similarity between two images and optionally display them"""
    # Load and preprocess
    img1 = keras.preprocessing.image.load_img(image_path1, target_size=(IMG_SIZE, IMG_SIZE))
    img2 = keras.preprocessing.image.load_img(image_path2, target_size=(IMG_SIZE, IMG_SIZE))
    
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
    
    # Display
    if show_images:
        plt.figure(figsize=(6, 3))
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
        plt.savefig('model_prediction.png')
        plt.show()
    
    return similarity, prediction, distance

# SIFT feature extraction and matching
def sift_match_and_show(image_path1, image_path2, num_matches=50):
    """Performs SIFT feature extraction, matches them, and shows the top matches."""
    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        print(f"Error: One or both image paths are invalid: {image_path1}, {image_path2}")
        return 0, 0

    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and extract features
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Draw keypoints
    def show_keypoints(image, keypoints, title):
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        img_with_kp = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 4))
        plt.imshow(img_with_kp)
        plt.title(title)
        plt.axis('off')
        plt.savefig(f"{title.replace(' ', '_')}.png")
        plt.show()

    show_keypoints(image1, keypoints1, "Image 1 Keypoints")
    show_keypoints(image2, keypoints2, "Image 2 Keypoints")

    # Match descriptors using brute force matcher
    if descriptors1 is None or descriptors2 is None:
        print("No SIFT descriptors found in one or both images")
        return 0, 0
        
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
    
    # Show matches
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title(f"SIFT: {num_matches} matches, similarity: {sift_similarity:.1f}%, average distance: {avg_distance:.2f}")
    plt.axis('off')
    plt.savefig('sift_matches.png')
    plt.show()
    
    return num_matches, avg_distance, sift_similarity

# Function to compare multiple image pairs
def compare_image_pairs(model, image_pairs, threshold=0.7):
    """Compare multiple pairs of images using both model and SIFT"""
    results = []
    
    for i, (img1, img2) in enumerate(image_pairs):
        print(f"\n=== Comparing Pair {i+1}: {os.path.basename(img1)} vs {os.path.basename(img2)} ===")
        
        # Check if files exist
        if not os.path.exists(img1):
            print(f"Error: Image not found: {img1}")
            continue
        if not os.path.exists(img2):
            print(f"Error: Image not found: {img2}")
            continue
            
        # Model prediction
        print("\nModel Prediction:")
        model_similarity, prediction, distance = predict_similarity(model, img1, img2, threshold)
        print(f"  Similarity: {model_similarity:.2f}%, Prediction: {prediction}, Distance: {distance:.4f}")
        
        # SIFT comparison
        print("\nSIFT Comparison:")
        num_matches, avg_distance, sift_similarity = sift_match_and_show(img1, img2)
        print(f"  Found {num_matches} matches with average distance {avg_distance:.2f}")
        print(f"  SIFT similarity: {sift_similarity:.2f}%")
        
        # Store results
        results.append({
            'pair': (img1, img2),
            'model': {
                'similarity': model_similarity,
                'prediction': prediction,
                'distance': distance
            },
            'sift': {
                'num_matches': num_matches,
                'avg_distance': avg_distance,
                'similarity': sift_similarity
            }
        })
    
    return results

# Print summary of results
def print_summary(results):
    """Print summary of all comparison results"""
    print("\n=== Summary of Results ===")
    print(f"{'Pair':<30} | {'Model Similarity':<20} | {'SIFT Matches':<15} | {'SIFT Similarity':<15}")
    print("-" * 85)
    
    for r in results:
        img1 = os.path.basename(r['pair'][0])
        img2 = os.path.basename(r['pair'][1])
        pair_name = f"{img1[:10]}...{img2[:10]}"
        
        model_sim = f"{r['model']['similarity']:.1f}% ({r['model']['prediction']})"
        sift_matches = f"{r['sift']['num_matches']}"
        sift_sim = f"{r['sift']['similarity']:.1f}%"
        
        print(f"{pair_name:<30} | {model_sim:<20} | {sift_matches:<15} | {sift_sim:<15}")

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate image similarity using trained model and SIFT")
    parser.add_argument("--model", default="image_similarity_model", help="Path to the trained model")
    parser.add_argument("--images", nargs="+", help="Paths to image pairs (even number required)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (default: 0.7)")
    parser.add_argument("--dir", help="Directory to search for test images")
    parser.add_argument("--examples", action="store_true", help="Run with example images from test directory")
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Determine which images to compare
    image_pairs = []
    
    if args.images:
        # If images are provided directly
        if len(args.images) % 2 != 0:
            print("Error: Number of images must be even to form pairs")
            return
        
        for i in range(0, len(args.images), 2):
            image_pairs.append((args.images[i], args.images[i+1]))
    
    elif args.dir:
        # If a directory is provided, find image files
        if not os.path.isdir(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            return
            
        image_files = []
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    image_files.append(os.path.join(root, file))
        
        # Take the first 10 images and pair them with the next 10
        if len(image_files) < 2:
            print(f"Error: Not enough images found in {args.dir}")
            return
            
        # Create pairs (1st with 2nd, 3rd with 4th, etc.)
        for i in range(0, min(10, len(image_files) - 1), 2):
            image_pairs.append((image_files[i], image_files[i+1]))
    
    elif args.examples:
        # Use example images from test directory
        test_dirs = [d for d in os.listdir('test') if os.path.isdir(os.path.join('test', d))]
        
        if not test_dirs:
            print("Error: No test directories found")
            return
            
        # Try to find some pairs within classes
        for test_dir in test_dirs[:2]:  # Use first two classes
            class_dir = os.path.join('test', test_dir)
            image_files = [f for f in os.listdir(class_dir) 
                          if os.path.isfile(os.path.join(class_dir, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if len(image_files) >= 2:
                # Same class comparison
                image_pairs.append((
                    os.path.join(class_dir, image_files[0]),
                    os.path.join(class_dir, image_files[1])
                ))
                
        # Add cross-class comparison if we have at least 2 classes
        if len(test_dirs) >= 2:
            dir1 = os.path.join('test', test_dirs[0])
            dir2 = os.path.join('test', test_dirs[1])
            
            img1 = [f for f in os.listdir(dir1) 
                   if os.path.isfile(os.path.join(dir1, f)) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                   
            img2 = [f for f in os.listdir(dir2) 
                   if os.path.isfile(os.path.join(dir2, f)) and 
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if img1 and img2:
                image_pairs.append((
                    os.path.join(dir1, img1[0]),
                    os.path.join(dir2, img2[0])
                ))
    
    else:
        # If no arguments provided, use some default test images if they exist
        default_test_paths = [
            './test/t1.jpg', './test/t2.jpg',
            './test/t3.jpg', './test/t4.jpg'
        ]
        
        # Check which files exist
        valid_paths = [p for p in default_test_paths if os.path.exists(p)]
        
        if len(valid_paths) >= 2:
            for i in range(0, len(valid_paths) - 1, 2):
                image_pairs.append((valid_paths[i], valid_paths[i+1]))
        else:
            print("Error: No default test images found. Please provide images using --images, --dir, or --examples.")
            print("Example usage:")
            print("  python evaluate_unknown_images.py --images img1.jpg img2.jpg")
            print("  python evaluate_unknown_images.py --dir ./my_test_images")
            print("  python evaluate_unknown_images.py --examples")
            return
    
    # Run comparison
    print(f"Comparing {len(image_pairs)} image pairs...")
    results = compare_image_pairs(model, image_pairs, args.threshold)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main() 