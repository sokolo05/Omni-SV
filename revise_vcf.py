#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd

def correct_vcf(vcf_path, csv_path, output_vcf_path):
    """
    Corrects structural variants in a single VCF file based on OmniSV predictions.
    """
    # 1. Load prediction results map: {Record_ID: Prediction}
    df = pd.read_csv(csv_path)
    prediction_dict = {str(row['Record_ID']): str(row['Prediction']) for _, row in df.iterrows()}

    stats = {
        "Total": 0,
        "Kept": 0,
        "Filtered_MATCH": 0,
        "Corrected_Type": 0,
        "Not_Found": 0
    }

    print(f"[*] Revising VCF: {os.path.basename(vcf_path)}")

    with open(vcf_path, 'r') as vcf_in, open(output_vcf_path, 'w') as vcf_out:
        for line in vcf_in:
            if line.startswith('#'):
                vcf_out.write(line)
                continue

            stats["Total"] += 1
            fields = line.strip().split('\t')
            
            chrom = fields[0]
            pos = fields[1]
            info = fields[7]

            # 2. Extract original SV details directly from the INFO field to determine standard Record_ID
            orig_type = 'OTHER'
            if 'SVTYPE=DEL' in info:
                orig_type = 'DEL'
            elif 'SVTYPE=INS' in info:
                orig_type = 'INS'

            sv_len = 'NA'
            if 'SVLEN=' in info:
                try:
                    # Parse SVLEN value and handle absolute values/lists
                    sv_len_part = [x for x in info.split(';') if x.startswith('SVLEN=')][0]
                    sv_len = str(abs(int(sv_len_part.split('=')[1].split(',')[0])))
                except Exception:
                    sv_len = 'NA'

            # Reconstruct the precise record ID matching the feature extraction rule: f"{chrom}.{pos}.{sv_type}.{sv_len}"
            record_id = f"{chrom}.{pos}.{orig_type}.{sv_len}"
            pred_type = prediction_dict.get(record_id, None)

            if pred_type is None:
                # Fallback to keep the record if the prediction metadata is missing
                vcf_out.write(line)
                stats["Not_Found"] += 1
                continue

            # 3. Execution of filtering and correction logic
            if pred_type == 'MATCH':
                # Filter out False Positives (MATCH classification results)
                stats["Filtered_MATCH"] += 1
                continue

            elif pred_type == orig_type:
                vcf_out.write(line)
                stats["Kept"] += 1

            else:
                # Correct mismatches between DEL and INS types dynamically
                stats["Corrected_Type"] += 1
                fields[4] = f"<{pred_type}>"
                fields[7] = info.replace(f"SVTYPE={orig_type}", f"SVTYPE={pred_type}")
                vcf_out.write('\t'.join(fields) + '\n')

    print(f"    - Total Input Records: {stats['Total']}")
    print(f"    - Filtered Records (MATCH): {stats['Filtered_MATCH']}")
    print(f"    - Type Corrections Applied: {stats['Corrected_Type']}")
    print(f"    - Unmatched Records (Skipped): {stats['Not_Found']}")
    print(f"    - Final Saved Records: {stats['Kept'] + stats['Corrected_Type']}")

def main():
    parser = argparse.ArgumentParser(description="Omni-SV VCF downstream refinement processor")
    # Synchronized single file arguments mapping to OmniSV.py pipeline configuration
    parser.add_argument('--vcf', type=str, required=True, help='Path to input raw VCF file')
    parser.add_argument('--csv', type=str, required=True, help='Path to predictions CSV file')
    parser.add_argument('--out', type=str, required=True, help='Path to save output refined VCF file')

    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    correct_vcf(args.vcf, args.csv, args.out)

if __name__ == '__main__':
    main()
