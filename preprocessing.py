import cv2
import os
import argparse

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def process_images(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        image = cv2.imread(os.path.join(input_dir, img_name), 0)  # Load as grayscale
        processed_img = apply_clahe(image)
        cv2.imwrite(os.path.join(output_dir, img_name), processed_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir)
