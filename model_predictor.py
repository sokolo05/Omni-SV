import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model_arch import OmniSVModel

class OmniSVInferenceDataset(Dataset):
    def __init__(self, record_ids, feat_dirs):
        self.record_ids = record_ids
        self.feat_dirs = feat_dirs

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        rid = self.record_ids[idx]
        
        x_syn = np.load(os.path.join(self.feat_dirs['syn'], f"{rid}.npy"))
        x_evo = np.load(os.path.join(self.feat_dirs['evo'], f"{rid}.npy"))
        x_omics = np.load(os.path.join(self.feat_dirs['omics'], f"{rid}.npy"))
        
        return rid, torch.from_numpy(x_syn).float(), \
                    torch.from_numpy(x_evo).float(), \
                    torch.from_numpy(x_omics).float()

def run_prediction(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference started on: {device}")

    all_files = os.listdir(args.syn_dir)
    record_ids = [f.replace(".npy", "") for f in all_files if f.endswith(".npy")]
    print(f"[*] Total records found for prediction: {len(record_ids)}")

    feat_dirs = {
        'syn': args.syn_dir,
        'evo': args.evo_dir,
        'omics': args.omics_dir
    }

    # 1. Corrected class instantiation
    model = OmniSVModel(d_model=args.d_model).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[*] Loaded model weights from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model path {args.model_path} not found.")
    
    model.eval()

    dataset = OmniSVInferenceDataset(record_ids, feat_dirs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = []
    class_map = {0: "DEL", 1: "INS", 2: "MATCH"} 

    with torch.no_grad():
        for rids, x_syn, x_evo, x_omics in tqdm(loader, desc="Predicting"):
            x_syn, x_evo, x_omics = x_syn.to(device), x_evo.to(device), x_omics.to(device)
            
            logits, _, _, _, _ = model(x_syn, x_evo, x_omics)
            
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            for i in range(len(rids)):
                results.append({
                    "Record_ID": rids[i],
                    "Prediction": class_map[preds[i].item()],
                    "Confidence": confidences[i].item(),
                    "Prob_DEL": probs[i][0].item(),
                    "Prob_INS": probs[i][1].item(),
                    "Prob_MATCH": probs[i][2].item()
                })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[+] Prediction results saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni-SV Inference Pipeline")
    
    parser.add_argument("--syn_dir", type=str, required=True)
    parser.add_argument("--evo_dir", type=str, required=True)
    parser.add_argument("--omics_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    
    # 2. Synchronized argument name with OmniSV.py pipeline configuration
    parser.add_argument("--output", type=str, required=True)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    run_prediction(args)
