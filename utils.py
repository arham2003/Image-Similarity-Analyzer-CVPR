import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances

def get_squared_distance_matrix(embeddings, diag_value=0):
    """
    Calculate squared Euclidean distance matrix between embeddings
    
    Args:
        embeddings: Array of embeddings with shape (n_samples, n_features)
        diag_value: Value to set on the diagonal (default: 0)
        
    Returns:
        Distance matrix with shape (n_samples, n_samples)
    """
    # Compute pairwise squared Euclidean distances
    distances = euclidean_distances(embeddings, squared=True)
    
    # Set diagonal values
    np.fill_diagonal(distances, diag_value)
    
    return distances

def normalize_image(image):
    """Normalize image to [0, 1] range"""
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for model prediction"""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return img, img_array

def get_embedding(model, img_array):
    """Get embedding from preprocessed image"""
    return model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0] 