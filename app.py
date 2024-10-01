from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from models.nested_unet import NestedUNet
from models.attention_unet import AttentionUNet
from utils import load_image, save_mask

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = load_image(await file.read()) 
    model = NestedUNet() 
    mask = predict(model, image)
    save_mask(mask, 'output/mask.png')
    return {"filename": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
