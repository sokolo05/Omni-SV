import os
import json
import argparse
import pandas as pd
from reconstructor import GenomicReconstructor
from spoa_consensus import SpoaEngine
from visual_encoder import VisualEncoder

'''
# Usage Example:
python data_main.py \
  --input ./demo/variants.txt \
  --bam ./demo/sample_alignment.bam \
  --ref ./demo/reference.fasta \
  --img_out ./output/images/ \
  --json_path ./output/sequences/sv_consensus.json \
  --cigar_ext 500 \
  --base_ext 256 \
  --select 30
'''
def main():
    parser = argparse.ArgumentParser(description="Modular SV Feature Extraction")
    # Input files
    parser.add_argument("-i", "--input", required=True, help="Input SV file")
    parser.add_argument("-b", "--bam", required=True, help="BAM path")
    parser.add_argument("-r", "--ref", required=True, help="Reference FASTA")
    
    # Output parameters
    parser.add_argument("--img_out", required=True, help="Directory for CIGAR PNGs")
    parser.add_argument("--json_path", required=True, help="Full path for consensus JSON file")
    
    # Tuning parameters
    parser.add_argument("--cigar_ext", type=int, default=500, help="CIGAR window size (total width = 2*ext+1)")
    parser.add_argument("--base_ext", type=int, default=256, help="Base consensus window size (total width = 2*ext+1)")
    parser.add_argument("-s", "--select", type=int, default=30, help="Max reads")

    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.img_out, exist_ok=True)
    if os.path.dirname(args.json_path):
        os.makedirs(os.path.dirname(args.json_path), exist_ok=True)

    recon = GenomicReconstructor(args.bam, args.ref)
    spoa_api = SpoaEngine()
    encoder = VisualEncoder()
    
    consensus_library = {}
    sep = '\t' if args.input.endswith(('.txt', '.tsv')) else ','
    df = pd.read_csv(args.input, sep=sep)

    for _, row in df.iterrows():
        chrom, pos = str(row['chr']), int(row['pos'])
        sv_type, sv_len = str(row.get('sv_type', 'NA')), str(row.get('sv_len', 'NA'))
        record_id = f"{chrom}.{pos}.{sv_type}.{sv_len}"
        
        # 1. Get reconstructed data for all reads at this position 
        # (Take the larger of the two extension lengths to ensure sufficient data)
        max_ext = max(args.cigar_ext, args.base_ext)
        raw_reads = recon.reconstruct_full_reads(chrom, pos, max_ext, args.select)
        
        if not raw_reads:
            print(f"Skipping {record_id}: No covering reads.")
            continue

        # 2. Process CIGAR matrix and generate images
        cigar_matrix = []
        for r in raw_reads:
            c, center = r['cigar'], r['center_idx']
            # Slice centered at the target position
            sliced_c = [c[i] if 0 <= i < len(c) else '-' 
                      for i in range(center - args.cigar_ext, center + args.cigar_ext + 1)]
            cigar_matrix.append(sliced_c)
        
        # Vertical padding
        while len(cigar_matrix) < args.select:
            cigar_matrix.append(['-'] * (2 * args.cigar_ext + 1))
        
        encoder.save_cigar_image(cigar_matrix, os.path.join(args.img_out, f"{record_id}.png"))

        # 3. Process Base sequences and perform SPOA consensus
        sliced_base_sequences = []
        for r in raw_reads:
            b, center = r['base'], r['center_idx']
            # Slice centered at the position to ensure coordinate alignment
            # Fill with 'N' if out of read range
            sliced_b = [b[i] if 0 <= i < len(b) else 'N' 
                      for i in range(center - args.base_ext, center + args.base_ext + 1)]
            sliced_base_sequences.append("".join(sliced_b))

        # Calculate consensus for these "local fragments"
        consensus_library[record_id] = spoa_api.generate_consensus(sliced_base_sequences)

    # 4. Save JSON
    with open(args.json_path, "w") as jf:
        json.dump(consensus_library, jf, indent=4)

    recon.close()
    print(f"Process complete. Images in {args.img_out}, JSON at {args.json_path}")

if __name__ == "__main__":
    main()
