import matplotlib.pyplot as plt
import numpy as np

def show_images(images, titles=None, cols=5, figsize=(15, 10)):
    """Display a list of images in a grid"""
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_similar_images(query_image, similar_images, distances=None, figsize=(15, 3)):
    """Show a query image and its similar images"""
    images = [query_image] + similar_images
    titles = ['Query Image'] + [f'Similar {i+1}' + (f': {d:.4f}' if distances else '') 
                                for i, d in enumerate(distances if distances else range(len(similar_images)))]
    
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show() 