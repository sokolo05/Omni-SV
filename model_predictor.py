import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import the architecture from your local file
from model_arch import OmniSVModel

# ==============================================================================
# INFERENCE DATASET
# ==============================================================================
class OmniSVInferenceDataset(Dataset):
    """
    Dataset for inference that maps record IDs to their triplet .npy files.
    """
    def __init__(self, record_ids, feat_dirs):
        self.record_ids = record_ids
        self.feat_dirs = feat_dirs

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        rid = self.record_ids[idx]
        
        # Load pre-processed features (Section 4.3 Representation)
        x_syn = np.load(os.path.join(self.feat_dirs['syn'], f"{rid}.npy"))
        x_evo = np.load(os.path.join(self.feat_dirs['evo'], f"{rid}.npy"))
        x_omics = np.load(os.path.join(self.feat_dirs['omics'], f"{rid}.npy"))
        
        return rid, torch.from_numpy(x_syn).float(), \
                    torch.from_numpy(x_evo).float(), \
                    torch.from_numpy(x_omics).float()

# ==============================================================================
# PREDICTION PIPELINE
# ==============================================================================
def run_prediction(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[*] Inference started on: {device}")

    # 1. Prepare record list (Assuming file names in syn_dir define the workload)
    all_files = os.listdir(args.syn_dir)
    record_ids = [f.replace(".npy", "") for f in all_files if f.endswith(".npy")]
    print(f"[*] Total records found for prediction: {len(record_ids)}")

    feat_dirs = {
        'syn': args.syn_dir,
        'evo': args.evo_dir,
        'omics': args.omics_dir
    }

    # 2. Initialize Model
    # Use d_model=256 as specified in paper Section 4.6
    model = OmniSVModel(d_model=args.d_model).to(device)
    
    # Load the best weights (e.g., the one saved with highest F1 score)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[*] Loaded model weights from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model path {args.model_path} not found.")
    
    model.eval()

    # 3. DataLoader for batch inference
    dataset = OmniSVInferenceDataset(record_ids, feat_dirs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 4. Inference Loop
    results = []
    class_map = {0: "DEL", 1: "INS", 2: "MATCH"} # Based on Section 4.6 labels

    with torch.no_grad():
        for rids, x_syn, x_evo, x_omics in tqdm(loader, desc="Predicting"):
            x_syn, x_evo, x_omics = x_syn.to(device), x_evo.to(device), x_omics.to(device)
            
            # Forward pass through Omni-SV
            # model_arch returns: logits, f_local, h_evo, f_final, h_omics
            logits, _, _, _, _ = model(x_syn, x_evo, x_omics)
            
            probs = torch.softmax(logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)
            
            # Store results
            for i in range(len(rids)):
                results.append({
                    "Record_ID": rids[i],
                    "Prediction": class_map[preds[i].item()],
                    "Confidence": confidences[i].item(),
                    "Prob_DEL": probs[i][0].item(),
                    "Prob_INS": probs[i][1].item(),
                    "Prob_MATCH": probs[i][2].item()
                })

    # 5. Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"[+] Prediction results saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni-SV High-Throughput Inference Script")
    
    # Path Arguments
    parser.add_argument("--syn_dir", type=str, required=True, help="Dir containing syntactic .npy files")
    parser.add_argument("--evo_dir", type=str, required=True, help="Dir containing evolutionary .npy files")
    parser.add_argument("--omics_dir", type=str, required=True, help="Dir containing functional .npy files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save prediction results CSV")
    
    # Model & Hardware Arguments
    parser.add_argument("--d_model", type=int, default=256, help="Hidden dimension (must match training)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for 4090 inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU threads for data loading")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU ID")

    args = parser.parse_args()
    run_prediction(args)