# Brain MRI Metastasis Segmentation

This project demonstrates the segmentation of brain metastases from MRI images using two advanced architectures: Nested U-Net and Attention U-Net.

## Features
- **CLAHE** preprocessing to enhance MRI images.
- **Nested U-Net** and **Attention U-Net** implementations for metastasis segmentation.
- A **FAST API** backend to serve the best-performing model.
- A **Streamlit UI** for easy image upload and visualization of segmentation results.

---

## Setup and Running Instructions
```bash
git clone https://github.com/sravani-kilari/5C-network.git
cd 5C-network
pip install -r requirements.txt
python preprocessing/preprocess.py --input_dir data/raw --output_dir data/preprocessed
python training/train.py --model nested_unet --epochs 50 --batch_size 16 --lr 0.001 --device cuda
python inference/inference.py --model_path path/to/nested_unet_weights.pth --model nested_unet --input_image path/to/input_image.png --output_image path/to/output_mask.png
python inference/inference.py --model_path path/to/attention_unet_weights.pth --model attention_unet --input_image path/to/input_image.png --output_image path/to/output_mask.png
uvicorn webapp.app:app --reload
streamlit run webapp/streamlit_ui.py

