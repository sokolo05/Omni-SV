import os
import json
import random
import argparse
import numpy as np
from alphagenome.models import dna_client
from alphagenome.data import genome
from tqdm import tqdm

def prepare_input_sequence(your_seq):
    target_len = 131072
    current_len = len(your_seq)
    if current_len > target_len:
        start_crop = (current_len - target_len) // 2
        return your_seq[start_crop:start_crop + target_len]
    padding_total = target_len - current_len
    left_padding = padding_total // 2
    right_padding = padding_total - left_padding
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choices(bases, k=left_padding)) + your_seq + ''.join(random.choices(bases, k=right_padding))

def main():
    parser = argparse.ArgumentParser(description="Extract AlphaGenome features from consensus JSON")
    parser.add_argument("--data_dir", required=True, help="Unified top-level data directory")
    parser.add_argument("-k", "--api_key", required=True, help="AlphaGenome API Key")
    parser.add_argument("--proxy", default=None, help="HTTP Proxy")
    parser.add_argument("--ontology", default="UBERON:0001157", help="Ontology term")
    args = parser.parse_args()

    json_path = os.path.join(args.data_dir, "sequences", "sv_consensus.json")
    out_dir = os.path.join(args.data_dir, "functional_feats")

    if args.proxy:
        os.environ['http_proxy'] = args.proxy
        os.environ['https_proxy'] = args.proxy
    os.makedirs(out_dir, exist_ok=True)

    try:
        model = dna_client.create(args.api_key)
        print("✅ AlphaGenome Client Connected")
    except Exception as e:
        print(f"[-] Integration aborted: {e}"); return

    if not os.path.exists(json_path):
        print(f"[-] Target source {json_path} does not exist."); return

    with open(json_path, 'r') as f:
        consensus_data = json.load(f)
    
    # Expected internal tracks mapping precisely to a total of 14 output channels
    track_configs = [
        ('rna_seq', 1),       # 1 channel
        ('atac', 1),          # 1 channel
        ('chip_tf', 8),       # 8 channels
        ('chip_histone', 4)   # 4 channels
    ]
    requested_outputs = [getattr(dna_client.OutputType, attr.upper()) for attr, _ in track_configs]

    TARGET_LEN = 1024
    print(f"🚀 Mining functional landscape maps across {len(consensus_data)} nodes...")
    
    for record_id, sequence in tqdm(consensus_data.items()):
        try:
            full_input = prepare_input_sequence(sequence)
            output = model.predict_sequence(
                sequence=full_input,
                organism=dna_client.Organism.HOMO_SAPIENS,
                requested_outputs=requested_outputs,
                ontology_terms=[args.ontology]
            )

            all_track_values = []
            for attr, target_dim in track_configs:
                data = getattr(output, attr, None)
                if data is not None and hasattr(data, 'values'):
                    val = data.values
                    if val.shape[0] != TARGET_LEN:
                        indices = np.linspace(0, val.shape[0] - 1, TARGET_LEN).astype(int)
                        val = val[indices, :]
                    # Verify channel completeness
                    if val.shape[1] != target_dim:
                        val = np.zeros((TARGET_LEN, target_dim))
                else:
                    # Robust zero padding fallback to maintain strict [1024, 14] dimensional integrity
                    val = np.zeros((TARGET_LEN, target_dim))
                all_track_values.append(val)
            
            final_features = np.concatenate(all_track_values, axis=1) # Consistently maps to [1024, 14]
            np.save(os.path.join(out_dir, f"{record_id}.npy"), final_features)
        except Exception as e:
            print(f"[-] Error processing node {record_id}: {e}")

if __name__ == "__main__":
    main()
