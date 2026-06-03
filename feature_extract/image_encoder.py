import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

class SpatialFeatureAligner(nn.Module):
    def __init__(self, in_channels=1280, target_len=1024, target_dim=512):
        super(SpatialFeatureAligner, self).__init__()
        # Project spatial tensor into the standardized shape [1024, 512]
        self.conv = nn.Conv1d(in_channels, target_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(target_len)

    def forward(self, x):
        # Input shape from backbone: [1, 1280, 7, 7]
        batch, c, h, w = x.shape
        x = x.view(batch, c, h * w) # [1, 1280, 49]
        x = self.conv(x)            # [1, 512, 49]
        x = self.pool(x)            # [1, 512, 1024]
        return x.transpose(1, 2)    # [1, 1024, 512]

def main():
    parser = argparse.ArgumentParser(description="Extract CIGAR image features using MobileNetV2")
    parser.add_argument("--data_dir", required=True, help="Unified top-level data directory")
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, "cigar_images")
    out_dir = os.path.join(args.data_dir, "syn_feats")
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize MobileNetV2 features extraction backbone
    backbone = models.mobilenet_v2(pretrained=True).features.to(device).eval()
    aligner = SpatialFeatureAligner(in_channels=1280, target_len=1024, target_dim=512).to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(img_dir):
        print(f"[-] Image directory {img_dir} missing."); return

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[*] Extracting features from {len(image_files)} syntactic images...")

    for filename in tqdm(image_files):
        record_id = os.path.splitext(filename)[0]
        save_path = os.path.join(out_dir, f"{record_id}.npy")
        
        try:
            img = Image.open(os.path.join(img_dir, filename)).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                raw_features = backbone(img_tensor)
                standardized_features = aligner(raw_features)
                features_np = standardized_features.squeeze(0).cpu().numpy() # Yields exactly [1024, 512]
            
            np.save(save_path, features_np)
        except Exception as e:
            print(f"[-] Error parsing {filename}: {e}")

if __name__ == "__main__":
    main()
