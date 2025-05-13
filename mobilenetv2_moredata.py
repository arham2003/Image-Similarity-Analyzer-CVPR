import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import cv2

# Set GPU memory growth and configuration
physical_devices = tf.config.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth enabled")
    
# CHANGE 1: Disable XLA optimization which is causing the error
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'

# Set environment variables for better GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_USE_CUDNN_BLAS_GEMM'] = '1'

# Set random seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

# Define data augmentation parameters - reduced intensity for better generalization
DATA_AUG_PARAMS = {
    'BRIGHTNESS': 0.3,
    'HUE': 0.3,
    'CONTRAST_MIN': 0.7,
    'CONTRAST_MAX': 1.3,
    'SATURATION_MIN': 0.7,
    'SATURATION_MAX': 1.3,
    'ZOOM_FACTOR': 0.3,
    'ROTATION_FACTOR': 0.15
}

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join('train'),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None,
    shuffle=True
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join("test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None,
    shuffle=False
)

# Create data augmentation pipeline
def augment_image(image):
    image = tf.image.random_brightness(image, DATA_AUG_PARAMS['BRIGHTNESS'])
    image = tf.image.random_contrast(image, DATA_AUG_PARAMS['CONTRAST_MIN'], DATA_AUG_PARAMS['CONTRAST_MAX'])
    image = tf.image.random_hue(image, DATA_AUG_PARAMS['HUE'])
    image = tf.image.random_saturation(image, DATA_AUG_PARAMS['SATURATION_MIN'], DATA_AUG_PARAMS['SATURATION_MAX'])
    image = tf.image.random_flip_left_right(image)
    
    # Apply random zoom
    random_zoom = tf.keras.layers.RandomZoom(DATA_AUG_PARAMS['ZOOM_FACTOR'])
    image = random_zoom(tf.expand_dims(image, 0))[0]
    
    # Apply random rotation
    random_rotation = tf.keras.layers.RandomRotation(DATA_AUG_PARAMS['ROTATION_FACTOR'])
    image = random_rotation(tf.expand_dims(image, 0))[0]
    
    return image

# CHANGE 2: Simplify dataset preparation to reduce GPU memory pressure
ds_train = (
    train_dataset
    .map(lambda x, y: (x / 255., y))
    .map(lambda x, y: (augment_image(x), y))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

ds_valid = (
    valid_dataset
    .map(lambda x, y: (x / 255., y))
    .batch(BATCH_SIZE * 2)
    .prefetch(tf.data.AUTOTUNE)
)

# Extract class labels for evaluation
train_images, train_labels = [], []
for images, labels in train_dataset.batch(100):
    train_images.append(images.numpy() / 255.0)
    train_labels.append(labels.numpy())
train_images = np.vstack(train_images)
train_labels = np.concatenate(train_labels)

valid_images, valid_labels = [], []
for images, labels in valid_dataset.batch(100):
    valid_images.append(images.numpy() / 255.0)
    valid_labels.append(labels.numpy())
valid_images = np.vstack(valid_images)
valid_labels = np.concatenate(valid_labels)

# Create updated model with regularization and dropout to prevent overfitting
def create_model():
    # CHANGE 3: Remove explicit device placement
    # Create the base MobileNetV2 model with pre-trained weights
    base_model = keras.applications.MobileNetV2(
        include_top=False, 
        pooling='avg', 
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )
    
    # Fine-tune only the top layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Build the model
    model = keras.Sequential([
        base_model,
        keras.layers.Dropout(0.3),  # Add dropout for regularization
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Dropout(0.2),  # Add another dropout layer
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))
    ])
    
    return model

# CHANGE 4: Remove mixed precision policy which may cause issues
# keras.mixed_precision.set_global_policy('mixed_float16')

# Create and compile the model
cnn = create_model()
cnn.compile(
    loss=tfa.losses.TripletSemiHardLoss(),  # Use semi-hard triplet loss for better stability
    optimizer=keras.optimizers.Adam(4.05e-5)
)


# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=12,  # Increased patience to allow LR adjustments to take effect
        min_delta=0.002,  # Minimum significant improvement
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive learning rate reduction
        patience=3,
        min_lr=1e-5,  # Lower minimum learning rate
        cooldown=1  # Add cooldown period after LR reduction
    ),
    keras.callbacks.TensorBoard(
        log_dir='./logs',
        update_freq='epoch'
    )
]

# CHANGE 5: Try-except block to gracefully handle GPU errors
try:
    # Train the model
    history = cnn.fit(
        ds_train,
        validation_data=ds_valid,
        callbacks=callbacks,
        epochs=EPOCHS,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Generate embeddings
    train_embeddings = cnn.predict(train_images)
    valid_embeddings = cnn.predict(valid_images)

except tf.errors.ResourceExhaustedError:
    print("GPU memory exhausted. Continuing with CPU...")
    # Fall back to CPU for prediction if needed
    with tf.device('/CPU:0'):
        train_embeddings = cnn.predict(train_images, batch_size=8)
        valid_embeddings = cnn.predict(valid_images, batch_size=8)

# Calculate accuracy using KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)
valid_pred = knn.predict(valid_embeddings)
accuracy = accuracy_score(valid_labels, valid_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

plt.subplot(1, 2, 2)
plt.bar(['Validation Accuracy'], [accuracy])
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Calculate similarity matrix
def get_squared_distance_matrix(embeddings, diag_value=np.inf):
    """Calculate pairwise squared Euclidean distance matrix between embeddings"""
    n = len(embeddings)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i, j] = diag_value  # Set diagonal to infinity
            else:
                distances[i, j] = np.sum((embeddings[i] - embeddings[j]) ** 2)
    
    return distances

d2 = get_squared_distance_matrix(valid_embeddings, diag_value=np.inf)

# Function to calculate similarity percentage
def calculate_similarity_percentage(distance):
    """Convert distance to similarity percentage (inverse relationship)"""
    # Lower distance means higher similarity
    # We'll use an exponential decay function: similarity = e^(-distance)
    similarity = np.exp(-distance)
    return similarity * 100  # Convert to percentage

# Visualize similar images with similarity percentages
def show_similar_images_with_percentages(images, distance_matrix, num_images=5, num_similar=3):
    """Show images with their most similar counterparts and similarity percentages"""
    plt.figure(figsize=(15, num_images * 3))
    
    # Randomly select indices to display
    indices = np.random.choice(len(images), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the distances to all other images
        distances = distance_matrix[idx]
        
        # Get indices of most similar images (excluding itself)
        similar_indices = np.argsort(distances)[:num_similar+1]
        similar_indices = similar_indices[similar_indices != idx][:num_similar]
        
        # Calculate similarity percentages
        similarities = [calculate_similarity_percentage(distances[j]) for j in similar_indices]
        
        # Display query image
        plt.subplot(num_images, num_similar + 1, i * (num_similar + 1) + 1)
        plt.imshow(images[idx])
        plt.title("Query Image")
        plt.axis('off')
        
        # Display similar images with similarity percentages
        for j, (sim_idx, sim_percent) in enumerate(zip(similar_indices, similarities)):
            plt.subplot(num_images, num_similar + 1, i * (num_similar + 1) + j + 2)
            plt.imshow(images[sim_idx])
            plt.title(f"Similarity: {sim_percent:.1f}%")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('similar_images.png')
    plt.show()

# Display similar images with similarity percentages
show_similar_images_with_percentages(valid_images, d2, num_images=5, num_similar=3)

# Function to predict similarity for new images
def predict_similarity(model, image_path1, image_path2):
    """Calculate similarity percentage between two images"""
    # Load and preprocess images
    img1 = keras.preprocessing.image.load_img(image_path1, target_size=(IMG_SIZE, IMG_SIZE))
    img2 = keras.preprocessing.image.load_img(image_path2, target_size=(IMG_SIZE, IMG_SIZE))
    
    img1 = keras.preprocessing.image.img_to_array(img1) / 255.0
    img2 = keras.preprocessing.image.img_to_array(img2) / 255.0
    
    # CHANGE 6: Error handling for prediction
    try:
        # Get embeddings
        embedding1 = model.predict(np.expand_dims(img1, axis=0), verbose=0)
        embedding2 = model.predict(np.expand_dims(img2, axis=0), verbose=0)
    except:
        print("Using CPU for prediction due to GPU error")
        with tf.device('/CPU:0'):
            embedding1 = model.predict(np.expand_dims(img1, axis=0), verbose=0)
            embedding2 = model.predict(np.expand_dims(img2, axis=0), verbose=0)
    
    # Calculate distance
    distance = np.sum((embedding1 - embedding2) ** 2)
    
    # Calculate similarity percentage
    similarity = calculate_similarity_percentage(distance)
    
    return similarity

# Save the model
cnn.save('image_similarity_model_4004')
print("Model saved successfully!")

# Load your saved model for inference
model = keras.models.load_model('image_similarity_model', compile=False)

# Function to calculate similarity percentage with visualization
def calculate_similarity_percentage(distance, scale=5.0):
    """Convert distance to a percentage similarity score"""
    similarity = np.exp(-distance / scale)
    return round(similarity * 100, 2)

def predict_similarity(model, image_path1, image_path2, threshold=0.7, show_images=True):
    """Predict similarity between two images and optionally display them"""
    # Load and preprocess
    img1 = keras.preprocessing.image.load_img(image_path1, target_size=(IMG_SIZE, IMG_SIZE))
    img2 = keras.preprocessing.image.load_img(image_path2, target_size=(IMG_SIZE, IMG_SIZE))
    
    img1_array = keras.preprocessing.image.img_to_array(img1) / 255.0
    img2_array = keras.preprocessing.image.img_to_array(img2) / 255.0
    
    # CHANGE 7: Error handling for prediction
    try:
        # Predict embeddings
        emb1 = model.predict(np.expand_dims(img1_array, axis=0), verbose=0)
        emb2 = model.predict(np.expand_dims(img2_array, axis=0), verbose=0)
    except:
        print("Using CPU for prediction due to GPU error")
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
        
        plt.suptitle(f"{prediction} ({distance:.2f} Distance)")
        plt.tight_layout()
        plt.savefig('prediction_result.png')
        plt.show()
    
    return similarity, prediction, distance

# Try to find test images
test_dirs = [d for d in os.listdir('test') if os.path.isdir(os.path.join('test', d))]
if test_dirs and len(test_dirs) > 0:
    first_class_dir = os.path.join('test', test_dirs[0])
    test_files = [f for f in os.listdir(first_class_dir) if os.path.isfile(os.path.join(first_class_dir, f))]
    if len(test_files) >= 2:
        test_image1 = os.path.join(first_class_dir, test_files[0])
        test_image2 = os.path.join(first_class_dir, test_files[2])
        
        similarity, prediction, distance = predict_similarity(
            model,
            test_image1,
            test_image2
        )
        print(f"Similarity: {similarity}%, Prediction: {prediction}, Distance: {distance:.4f}")

# SIFT feature extraction and matching
def sift_match_and_show(image_path1, image_path2, num_matches=50):
    """Performs SIFT feature extraction, matches them, and shows the top matches."""

    # Load images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        print("Error: One or both image paths are invalid.")
        return

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
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    matched_img = cv2.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches[:num_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    # Show matches
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.title(f"Top {num_matches} Feature Matches")
    plt.axis('off')
    plt.savefig('sift_matches.png')
    plt.show()
    
    # Return match information
    avg_distance = np.mean([m.distance for m in matches[:num_matches]]) if matches else float('inf')
    return len(matches), avg_distance

# Perform SIFT matching on test images if available
if 'test_image1' in locals() and 'test_image2' in locals():
    print("\nComparing images with SIFT:")
    num_matches, avg_distance = sift_match_and_show(test_image1, test_image2)
    print(f"Found {num_matches} matches with average distance {avg_distance:.2f}")

    print("\nMore images with SIFT:")
    num_matches, avg_distance = sift_match_and_show('./test/t1.jpg', './test/t2.jpg')
    print(f"Found {num_matches} matches with average distance {avg_distance:.2f}")

print("\nAll processing complete!")