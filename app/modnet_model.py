import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from io import BytesIO
import cv2

from app.modnet.modnet import MODNet

# Define hyper-parameters
ref_size = 1024  # Increased from 512 to 1024 for better quality

# Define image to tensor transform with better quality preservation
im_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    ]
)

# Function to load the model
def load_model(ckpt_path: str):
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet

# Function to remove the background and save it as PNG
def remove_background(image_bytes: bytes, model: nn.Module):
    # Open and convert image to RGB mode
    im = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Convert image to numpy and ensure it's 3-channel
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # Store original image dimensions
    original_h, original_w = im.shape[:2]

    # Convert image to tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # Add mini-batch dimension
    im = im[None, :, :, :]

    # Resize image for input while maintaining aspect ratio
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w
    
    # Ensure dimensions are multiples of 32 for the model
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    
    # Use bicubic interpolation for better quality
    im = F.interpolate(im, size=(im_rh, im_rw), mode='bicubic', align_corners=True)

    # Inference to get the matte (background mask)
    _, _, matte = model(im.cuda() if torch.cuda.is_available() else im, True)

    # Resize matte to match original image size using bicubic interpolation
    matte = F.interpolate(matte, size=(original_h, original_w), mode='bicubic', align_corners=True)
    matte = matte[0][0].data.cpu().numpy()

    # Convert input image back to numpy and rescale
    im = np.asarray(im[0].cpu().detach().numpy().transpose(1, 2, 0))
    im = (im * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255  # Denormalize
    im = np.clip(im, 0, 255).astype(np.uint8)

    # Resize the processed image back to original dimensions using high-quality resampling
    im = Image.fromarray(im)
    im = im.resize((original_w, original_h), Image.Resampling.LANCZOS)
    im = np.asarray(im)

    # Apply Gaussian blur to the matte for smoother edges
    matte = (matte > 0.5).astype(np.uint8)
    matte = cv2.GaussianBlur(matte, (5, 5), 0)

    # Apply the matte mask to the image with improved edge handling
    result = im * matte[:, :, None]

    # Add alpha channel with improved edge handling
    alpha = matte * 255
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    result_rgba = np.dstack((result, alpha))

    # Save the result as a PNG with maximum quality
    result_image = Image.fromarray(result_rgba.astype(np.uint8), mode='RGBA')
    return result_image
