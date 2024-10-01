import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    return torch.from_numpy(image).float().unsqueeze(0) 

def save_mask(mask, output_path):
    mask = (mask > 0.5).numpy().astype(np.uint8) * 255 
    cv2.imwrite(output_path, mask)
