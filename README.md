# Brain MRI Metastasis Segmentation

This project demonstrates the segmentation of brain metastases from MRI images using two advanced architectures: Nested U-Net and Attention U-Net.

## Features
- **CLAHE** preprocessing to enhance MRI images.
- **Nested U-Net** and **Attention U-Net** implementations for metastasis segmentation.
- A **FAST API** backend to serve the best-performing model.
- A **Streamlit UI** for easy image upload and visualization of segmentation results.

---

## Setup and Running Instructions

### Step 1: Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/sravani-kilari/5C-network.git
cd 5C-network
pip install -r requirements.txt
