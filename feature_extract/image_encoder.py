import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

def build_backbone(model_name, freeze_layers=5):
    """Refined backbone builder based on your MobileNetV2 logic."""
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        layers = list(model.features.children())
        # Freeze layers
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        model.classifier = nn.Identity()
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Extract CIGAR image features using MobileNetV2")
    # Paths
    parser.add_argument("-i", "--img_dir", required=True, help="Directory containing CIGAR .png images")
    parser.add_argument("-o", "--out_dir", required=True, help="Directory to save .npy feature files")
    # Model Params
    parser.add_argument("--model_name", default="mobilenet_v2", help="Backbone model name")
    parser.add_argument("--freeze", type=int, default=5, help="Number of layers to freeze")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    model = build_backbone(args.model_name, args.freeze).to(device)
    model.eval()

    # 2. Define Transform
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Process Images
    image_files = [f for f in os.listdir(args.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[*] Found {len(image_files)} images. Extracting features...")

    for filename in tqdm(image_files):
        record_id = os.path.splitext(filename)[0]
        save_path = os.path.join(args.out_dir, f"{record_id}.npy")
        
        try:
            img = Image.open(os.path.join(args.img_dir, filename)).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model.features(img_tensor)
                features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features_np = torch.flatten(features, 1).squeeze(0).cpu().numpy()
            
            np.save(save_path, features_np)
        except Exception as e:
            print(f"Error on {filename}: {e}")

if __name__ == "__main__":
    main()