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
    # Paths
    parser.add_argument("-j", "--json_path", required=True, help="Path to sv_consensus.json")
    parser.add_argument("-o", "--out_dir", required=True, help="Directory to save .npy files")
    parser.add_argument("-k", "--api_key", required=True, help="AlphaGenome API Key")
    # Config
    parser.add_argument("--proxy", default=None, help="HTTP Proxy (e.g., http://127.0.0.1:7890)")
    parser.add_argument("--ontology", default="UBERON:0001157", help="Ontology term")
    args = parser.parse_args()

    # 1. Setup Env
    if args.proxy:
        os.environ['http_proxy'] = args.proxy
        os.environ['https_proxy'] = args.proxy
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. Init Client
    try:
        model = dna_client.create(args.api_key)
        print("✅ AlphaGenome Client Ready")
    except Exception as e:
        print(f"❌ Init Failed: {e}"); return

    # 3. Load Data
    with open(args.json_path, 'r') as f:
        consensus_data = json.load(f)
    
    requested_outputs = [
        dna_client.OutputType.RNA_SEQ, dna_client.OutputType.ATAC,
        dna_client.OutputType.CHIP_TF, dna_client.OutputType.CHIP_HISTONE
    ]

    # 4. Loop
    TARGET_LEN = 1024
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
            for attr in ['rna_seq', 'atac', 'chip_tf', 'chip_histone']:
                data = getattr(output, attr, None)
                if data is not None and hasattr(data, 'values'):
                    val = data.values
                    if val.shape[0] != TARGET_LEN:
                        indices = np.linspace(0, val.shape[0] - 1, TARGET_LEN).astype(int)
                        val = val[indices, :] 
                    all_track_values.append(val)
            
            if all_track_values:
                np.save(os.path.join(args.out_dir, f"{record_id}.npy"), np.concatenate(all_track_values, axis=1))
        except Exception as e:
            print(f"Error on {record_id}: {e}")

if __name__ == "__main__":
    main()