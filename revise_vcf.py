import pandas as pd
import argparse
import os

def correct_vcf(vcf_path, csv_path, output_vcf_path, conf_threshold=0.0):
    """
    Corrects VCF based on Omni-SV model predictions (DEL, INS, MATCH).
    """
    # 1. Load Prediction CSV
    # Expected columns: Record_ID, Prediction, Confidence
    df = pd.read_csv(csv_path)
    
    # Create a lookup dictionary: {Record_ID: Prediction}
    # Record_ID format usually: SampleID_Chr_Pos
    prediction_dict = {row['Record_ID']: row['Prediction'] for _, row in df.iterrows()}

    # Counters for statistics
    stats = {
        "Total": 0,
        "Kept": 0,
        "Filtered_MATCH": 0,
        "Corrected_Type": 0,
        "Not_Found": 0
    }

    print(f"[*] Processing: {os.path.basename(vcf_path)}")

    with open(vcf_path, 'r') as vcf_in, open(output_vcf_path, 'w') as vcf_out:
        for line in vcf_in:
            if line.startswith('#'):
                vcf_out.write(line)
                continue

            stats["Total"] += 1
            fields = line.strip().split('\t')
            
            # Standardizing chromosome and position
            chr_name = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            info = fields[7]

            # Construct Record_ID to match CSV (Logic should match your feature extraction)
            # Example: HG002_chr1_123456
            sample_name = os.path.basename(csv_path).replace('.csv', '')
            record_id = f"{sample_name}_{chr_name}_{pos}"

            # Get model prediction
            pred_type = prediction_dict.get(record_id, None)

            if pred_type is None:
                # If no prediction found, keep original entry or skip
                vcf_out.write(line)
                stats["Not_Found"] += 1
                continue

            # Identify original VCF type
            if 'SVTYPE=DEL' in info:
                orig_type = 'DEL'
            elif 'SVTYPE=INS' in info:
                orig_type = 'INS'
            else:
                orig_type = 'OTHER'

            # --- CORRECTION LOGIC ---
            
            # Case 1: Model predicts MATCH (False Positive)
            if pred_type == 'MATCH':
                stats["Filtered_MATCH"] += 1
                # We skip writing this line to effectively filter the false positive
                continue

            # Case 2: Type matches original
            elif pred_type == orig_type:
                vcf_out.write(line)
                stats["Kept"] += 1

            # Case 3: Type mismatch (Correcting DEL to INS or vice-versa)
            else:
                stats["Corrected_Type"] += 1
                new_alt = f"<{pred_type}>"
                new_info = info.replace(f"SVTYPE={orig_type}", f"SVTYPE={pred_type}")
                
                fields[4] = new_alt
                fields[7] = new_info
                vcf_out.write('\t'.join(fields) + '\n')

    # Print Summary for this file
    print(f"    - Total Records: {stats['Total']}")
    print(f"    - Filtered (MATCH): {stats['Filtered_MATCH']}")
    print(f"    - Corrected Type: {stats['Corrected_Type']}")
    print(f"    - Final Records: {stats['Kept'] + stats['Corrected_Type']}")

def main():
    parser = argparse.ArgumentParser(description="Omni-SV VCF Revision Tool")
    parser.add_argument('--vcf_dir', type=str, required=True, help='Input VCF folder')
    parser.add_argument('--csv_dir', type=str, required=True, help='Prediction CSV folder')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder')
    parser.add_argument('--conf_threshold', type=float, default=0.0, help='Minimum confidence to apply correction')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    vcf_files = [f for f in os.listdir(args.vcf_dir) if f.endswith('.vcf')]
    
    for vcf_file in vcf_files:
        # Match SampleID.vcf with SampleID.csv
        sample_id = vcf_file.replace('.all.vcf', '').replace('.vcf', '')
        csv_file = f"{sample_id}.csv"
        csv_path = os.path.join(args.csv_dir, csv_file)

        if os.path.exists(csv_path):
            vcf_path = os.path.join(args.vcf_dir, vcf_file)
            output_vcf_path = os.path.join(args.output_dir, vcf_file.replace('.vcf', '.omnisv_revised.vcf'))
            correct_vcf(vcf_path, csv_path, output_vcf_path, args.conf_threshold)
        else:
            print(f"[!] Warning: No prediction CSV found for {vcf_file}. Expected: {csv_file}")

if __name__ == '__main__':
    main()