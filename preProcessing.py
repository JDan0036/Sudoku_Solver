import math 
import json
import os 
from pathlib import Path 
import numpy as np 
from matplotlib import pyplot as plt 
import time 
from PIL import Image, ImageFilter

def adaptive_threshold(img, block_size=15, C=5):
    h, w = img.shape
    out = np.zeros_like(img)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img[y:y+block_size, x:x+block_size]
            thresh = np.mean(block) - C
            out[y:y+block_size, x:x+block_size] = block < thresh

    return (out.astype(np.uint8) * 255)

def preprocess_image(image):
    """
    Preprocess the given image by converting it to grayscale and applying binary thresholding.
    
    Args:
        image: The input image in cv2 format.
    """
    img = Image.open(image)
    img = img.resize((500, 500))
    gray = img.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    gray_np = np.array(blurred)
    gray_np = (gray_np - gray_np.min()) / (gray_np.max() - gray_np.min())
    gray_np = (gray_np * 255).astype(np.uint8)
    threshold = np.mean(gray_np)
    binary = adaptive_threshold(gray_np)

    plt.figure(figsize=(5,5))
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    plt.show()

    


preprocess_image("Full puzzle2.jpeg")