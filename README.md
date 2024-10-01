# Brain MRI Metastasis Segmentation

This project demonstrates the segmentation of brain metastases from MRI images using Nested U-Net and Attention U-Net models.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/sravani-kilari/5C-network.git
   cd 5C-network
pip install -r requirements.txt
python preprocessing/preprocess.py --input_dir data/raw --output_dir data/preprocessed
python training/train.py --model nested_unet --epochs 50 --batch_size 16 --lr 0.001 --device cuda
uvicorn webapp.app:app --reload
streamlit run webapp/streamlit_ui.py

