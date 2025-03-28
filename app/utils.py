# app/utils.py
import cv2
import numpy as np

def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    
def read_image(image_path):
    return cv2.imread(image_path)
