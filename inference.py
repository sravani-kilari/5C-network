import torch
import argparse
from models.nested_unet import NestedUNet
from models.attention_unet import AttentionUNet
from utils import load_image, save_mask  

def predict(model, image, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  
        output = model(image)
        return output.squeeze(0).cpu()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, choices=['nested_unet', 'attention_unet'], required=True)
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--output_image', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'nested_unet':
        model = NestedUNet()
    else:
        model = AttentionUNet()

    model.load_state_dict(torch.load(args.model_path))

    image = load_image(args.input_image)
    mask = predict(model, image, device)
    save_mask(mask, args.output_image)
