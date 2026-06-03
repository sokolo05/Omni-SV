import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from evo2 import Evo2

os.environ["NVTE_ALGO_USAGE_TRACKING"] = "0"

def resample_features(features, target_len=1024):
    current_len = features.shape[0]
    if current_len == target_len:
        return features
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    return features[indices, :]

def main():
    parser = argparse.ArgumentParser(description="Extract Evo2 features from consensus JSON")
    parser.add_argument("--data_dir", required=True, help="Unified top-level data directory")
    parser.add_argument("--model", default="evo2_1b_base", help="Evo2 model version")
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "sequences", "sv_consensus.json")
    out_dir = os.path.join(args.data_dir, "evo_feats")
    os.makedirs(out_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[*] Loading model {args.model} onto {device}...")
    try:
        model = Evo2(args.model)
        model.to(device).eval()
        
        # Determine the hidden state projector based on model architecture configuration
        mock_input = torch.zeros((1, 10), dtype=torch.long, device=device)
        with torch.no_grad():
            _, mock_embed = model(mock_input)
            actual_hidden_dim = mock_embed.shape[-1]
        
        # Hard alignment to fit the strict 2048 hidden representation expected by OmniSV
        embed_projector = nn.Linear(actual_hidden_dim, 2048).to(device).eval() if actual_hidden_dim != 2048 else None
    except Exception as e:
        print(f"[-] Model loading failed: {e}"); return

    if not os.path.exists(json_path):
        print(f"[-] Consensus path {json_path} missing."); return

    with open(json_path, 'r') as f:
        consensus_data = json.load(f)

    print(f"🚀 Processing {len(consensus_data)} evolutionary sequences...")
    for record_id, sequence in tqdm(consensus_data.items()):
        try:
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    input_ids = torch.tensor(
                        model.tokenizer.tokenize(sequence),
                        dtype=torch.long,
                    ).unsqueeze(0).to(device)
                    
                    logits, embeddings = model(input_ids)
                    
                    logits_np = logits.squeeze(0).cpu().float().numpy()
                    
                    if embed_projector is not None:
                        embeddings = embed_projector(embeddings.float())
                    embeds_np = embeddings.squeeze(0).cpu().float().numpy()

                    # Restrict vocabulary dimension to exactly 4 channels (A, C, G, T)
                    if logits_np.shape[-1] > 4:
                        logits_np = logits_np[:, :4]

                    combined_features = np.concatenate([logits_np, embeds_np], axis=-1)
                    combined_features = resample_features(combined_features, target_len=1024) # Force [1024, 2052]

                    np.save(os.path.join(out_dir, f"{record_id}.npy"), combined_features)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[-] Failure at node {record_id}: {e}")

if __name__ == "__main__":
    main()
