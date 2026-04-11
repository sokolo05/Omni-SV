import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from evo2 import Evo2

# ==============================================================================
# HARDWARE COMPATIBILITY LAYER
# ==============================================================================
os.environ["NVTE_ALGO_USAGE_TRACKING"] = "0"

def resample_features(features, target_len=1024):
    """
    Resample the sequence length to target_len (e.g., 1024).
    features shape: (Original_Len, Dim)
    """
    current_len = features.shape[0]
    if current_len == target_len:
        return features
    
    # Linear interpolation indexing for resampling
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    return features[indices, :]

def main():
    parser = argparse.ArgumentParser(description="Extract Evo2 features (Logits & Embeddings) from consensus JSON")
    # Path arguments
    parser.add_argument("-j", "--json_path", required=True, help="Path to sv_consensus.json")
    parser.add_argument("-o", "--out_dir", required=True, help="Directory to save .npy files")
    # Model arguments
    parser.add_argument("--model", default="evo2_1b_base", help="Evo2 model version")
    parser.add_argument("--resample", action="store_true", help="Resample length to 1024 (AlphaGenome style)")
    args = parser.parse_args()

    # 1. Environment Preparation
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Initialize Evo2
    print(f"[*] Initializing {args.model}...")
    try:
        # Utilizing RTX 4090 performance with Evo2
        model = Evo2(args.model)
        model.to(device).eval()
        print("✅ Evo2 Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Load Data
    with open(args.json_path, 'r') as f:
        consensus_data = json.load(f)
    print(f"📋 Loaded {len(consensus_data)} sequences from JSON")

    # 4. Extraction Loop
    print(f"🚀 Starting extraction on {device}...")
    for record_id, sequence in tqdm(consensus_data.items()):
        try:
            with torch.no_grad():
                # Optimized for 4090 using bfloat16 for speed and precision
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Tokenization
                    input_ids = torch.tensor(
                        model.tokenizer.tokenize(sequence),
                        dtype=torch.long,
                    ).unsqueeze(0).to(device)
                    
                    # Inference: Retrieve both Logits and Embeddings
                    # logits: [Batch, Len, Vocab]
                    # embeddings: [Batch, Len, Hidden_Dim]
                    logits, embeddings = model(input_ids)
                    
                    # Remove batch dimension and convert to float32 CPU numpy
                    # Note: We keep the full sequence length (Len) to support EGR module
                    logits_np = logits.squeeze(0).cpu().float().numpy()
                    embeds_np = embeddings.squeeze(0).cpu().float().numpy()

                    # Concatenate features along the last dimension
                    # Result shape: [Len, Vocab + Hidden_Dim]
                    # Corresponds to Section 4.3.2: E (embeddings) and L (logits)
                    combined_features = np.concatenate([logits_np, embeds_np], axis=-1)

                    # Optional resampling (if fixed length differs from target)
                    if args.resample:
                        combined_features = resample_features(combined_features, target_len=1024)

                    # Save as .npy file
                    save_path = os.path.join(args.out_dir, f"{record_id}.npy")
                    np.save(save_path, combined_features)

            # Clean cache for 24GB VRAM optimization
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n❌ Error processing {record_id}: {e}")
            continue

    print(f"\n✨ Evo2 Extraction Complete. Features saved to: {args.out_dir}")

if __name__ == "__main__":
    main()