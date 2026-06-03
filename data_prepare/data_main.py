#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import pandas as pd
from reconstructor import GenomicReconstructor
from spoa_consensus import SpoaEngine
from visual_encoder import VisualEncoder

def main():
    parser = argparse.ArgumentParser(description="Multi-modal Genomic SV Feature Preprocessing")
    
    # Standardized pipeline arguments perfectly matching OmniSV.py
    parser.add_argument("--vcf", required=True, help="Input structural variant file (VCF, TXT, or CSV format)")
    parser.add_argument("--bam", required=True, help="Path to the aligned BAM file")
    parser.add_argument("--ref", required=True, help="Path to the reference genome FASTA file")
    parser.add_argument("--out", required=True, help="Unified top-level output directory for downstream encoders")
    
    # Optimized window dimensions adjusted for GFM embeddings and 1024 matrix sizing
    parser.add_argument("--cigar_ext", type=int, default=511, help="CIGAR extension radius (Yields exact width of 2*511+1 = 1024)")
    parser.add_argument("--base_ext", type=int, default=1000, help="Base consensus window radius (Yields exact 2000bp region for GFM)")
    parser.add_argument("-s", "--select", type=int, default=30, help="Maximum coverage reads to sample per locus")

    args = parser.parse_args()
    
    # Automatically setup sub-directories inside the unified output folder
    img_out_dir = os.path.join(args.out, "cigar_images")
    json_out_dir = os.path.join(args.out, "sequences")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)
    
    json_path = os.path.join(json_out_dir, "sv_consensus.json")

    recon = GenomicReconstructor(args.bam, args.ref)
    spoa_api = SpoaEngine()
    encoder = VisualEncoder()
    
    consensus_library = {}
    sep = '\t' if args.vcf.endswith(('.txt', '.tsv', '.vcf')) else ','
    df = pd.read_csv(args.vcf, sep=sep)

    for _, row in df.iterrows():
        chrom, pos = str(row['chr']), int(row['pos'])
        sv_type, sv_len = str(row.get('sv_type', 'NA')), str(row.get('sv_len', 'NA'))
        record_id = f"{chrom}.{pos}.{sv_type}.{sv_len}"
        
        # Capture the maximum required flanking buffer across modalities
        max_ext = max(args.cigar_ext, args.base_ext)
        raw_reads = recon.reconstruct_full_reads(chrom, pos, max_ext, args.select)
        
        if not raw_reads:
            print(f"Skipping {record_id}: No spanning reads discovered.")
            continue

        # Extract and compile CIGAR alignments synchronized to the center coordinate
        cigar_matrix = []
        for r in raw_reads:
            c, center = r['cigar'], r['center_idx']
            sliced_c = [c[i] if 0 <= i < len(c) else '-' 
                      for i in range(center - args.cigar_ext, center + args.cigar_ext + 1)]
            cigar_matrix.append(sliced_c)
        
        while len(cigar_matrix) < args.select:
            cigar_matrix.append(['-'] * (2 * args.cigar_ext + 1))
        
        encoder.save_cigar_image(cigar_matrix, os.path.join(img_out_dir, f"{record_id}.png"))

        # Slice base sequences for local multi-sequence MSA consolidation
        sliced_base_sequences = []
        for r in raw_reads:
            b, center = r['base'], r['center_idx']
            sliced_b = [b[i] if 0 <= i < len(b) else 'N' 
                      for i in range(center - args.base_ext, center + args.base_ext + 1)]
            sliced_base_sequences.append("".join(sliced_b))

        consensus_library[record_id] = spoa_api.generate_consensus(sliced_base_sequences)

    with open(json_path, "w") as jf:
        json.dump(consensus_library, jf, indent=4)

    recon.close()
    print(f"[+] Execution finalized. Images placed in {img_out_dir}, Consensus JSON exported to {json_path}")

if __name__ == "__main__":
    main()
