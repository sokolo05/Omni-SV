import pandas as pd
import pysam
import random
import argparse
import sys
from collections import defaultdict

'''
# Usage:
python NIRBS_Algorithm.py \
    --sv_file input_svs.txt \
    --bam_file alignment.bam \
    --output final_dataset.tsv \
    --ratio 0.8 \
    --buffer 500
'''

def calculate_large_indel_ratio(bam_file, chrom, start, end, min_indel_length=30):
    """
    Calculates the ratio of reads containing I/D operations > 30bp in a specific region.
    """
    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
    except Exception as e:
        print(f"Error opening BAM file: {e}")
        return 1.0
        
    large_indel_reads = 0
    total_reads = 0
    
    try:
        # Use fetch to scan the specific genomic interval
        for read in bam.fetch(chrom, start, end):
            total_reads += 1
            has_large_indel = False
            if read.cigartuples:
                for op, length in read.cigartuples:
                    # Check for Insertions (1) or Deletions (2)
                    if length >= min_indel_length and (op == 1 or op == 2):
                        has_large_indel = True
                        break
            if has_large_indel:
                large_indel_reads += 1
    except ValueError:
        return 1.0
    finally:
        bam.close()
    
    if total_reads == 0:
        return 0.0
    
    return large_indel_reads / total_reads

def intervals_overlap(start1, end1, start2, end2):
    """Checks if two genomic intervals overlap."""
    return not (end1 < start2 or start1 > end2)

def check_position_in_exclusion_zones(pos, exclusion_zones):
    """Checks if a position falls within any exclusion zones."""
    for zone_start, zone_end in exclusion_zones:
        if zone_start <= pos <= zone_end:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Generate MATCH regions following SV length distribution.")
    
    # File Arguments
    parser.add_argument("-s", "--sv_file", required=True, help="Path to input SV TXT/TSV file")
    parser.add_argument("-b", "--bam_file", required=True, help="Path to input BAM file")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output file")
    
    # Logic Arguments
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio of MATCH records to generate relative to SV count (default: 0.8)")
    parser.add_argument("--buffer", type=int, default=3000, help="Buffer size around SVs and MATCHes (default: 3000)")
    parser.add_argument("--max_ratio", type=float, default=0.1, help="Max allowed large indel ratio in BAM (default: 0.1)")
    
    args = parser.parse_args()

    # 1. Load SV Data
    try:
        df = pd.read_csv(args.sv_file, sep="\t")
    except Exception as e:
        print(f"Error reading SV file: {e}")
        sys.exit(1)

    if 'sv_len' not in df.columns:
        df['sv_len'] = (df['sv_end'] - df['sv_start']).abs()

    global_length_pool = df['sv_len'].tolist()

    # 2. Get Chromosome Lengths from BAM
    chrom_lengths = {}
    try:
        bam = pysam.AlignmentFile(args.bam_file, "rb")
        chrom_lengths = {ref: length for ref, length in zip(bam.references, bam.lengths)}
        bam.close()
    except Exception as e:
        print(f"Error reading BAM header: {e}")
        sys.exit(1)

    summary = df.groupby(["chr", "sv_type"]).size().unstack(fill_value=0)
    match_records = []
    
    # 3. Process each chromosome
    for chrom in df["chr"].unique():
        chrom_str = str(chrom)
        if chrom_str not in chrom_lengths:
            print(f"Warning: Chromosome {chrom} not found in BAM. Skipping.")
            continue
        
        chrom_len = chrom_lengths[chrom_str]
        current_chrom_svs = df[df["chr"] == chrom]
        
        total_sv = len(current_chrom_svs)
        num_target = int(total_sv * args.ratio)
        
        if num_target == 0:
            continue

        # Build exclusion zones (SVs + Buffer)
        sv_exclusion_zones = []
        local_length_pool = current_chrom_svs['sv_len'].tolist()

        for _, row in current_chrom_svs.iterrows():
            e_start, e_end = row["sv_start"], row["sv_end"]
            if row["sv_type"] == "INS":
                e_end = max(e_end, row["sv_start"] + row["sv_len"])
            
            sv_exclusion_zones.append((max(1, e_start - args.buffer), min(chrom_len, e_end + args.buffer)))
        
        sv_exclusion_zones.sort()

        # Generation Loop
        gen_exclusion_zones = []
        generated = 0
        attempts = 0
        max_attempts = num_target * 300

        print(f"Processing {chrom}: Target {num_target} MATCHes...")

        while generated < num_target and attempts < max_attempts:
            attempts += 1
            
            # Sample length from distribution
            target_len = random.choice(local_length_pool if local_length_pool else global_length_pool)
            pos = random.randint(1, chrom_len)
            
            m_start = max(1, pos - (target_len // 2))
            m_end = min(chrom_len, m_start + target_len)
            
            # Check overlap with SVs
            if any(intervals_overlap(m_start, m_end, z[0], z[1]) for z in sv_exclusion_zones):
                continue
            
            # Check overlap with already generated MATCHes
            if any(intervals_overlap(m_start, m_end, z[0], z[1]) for z in gen_exclusion_zones):
                continue

            # Validate BAM signal
            ratio = calculate_large_indel_ratio(args.bam_file, chrom_str, m_start, m_end)
            if ratio >= args.max_ratio:
                continue

            # Record success
            match_id = f"MATCH_{chrom}_{generated:04d}"
            match_records.append([chrom, match_id, pos, "MATCH", target_len, m_start, m_end])
            
            # Add to exclusion to prevent MATCH-to-MATCH overlap
            gen_exclusion_zones.append((max(1, m_start - args.buffer), min(chrom_len, m_end + args.buffer)))
            generated += 1

        print(f"  Done: Created {generated}/{num_target} regions.")

    # 4. Save Results
    if match_records:
        df_match = pd.DataFrame(match_records, columns=["chr", "id", "pos", "sv_type", "sv_len", "sv_start", "sv_end"])
        pd.concat([df, df_match], ignore_index=True).to_csv(args.output, sep="\t", index=False)
        print(f"\nFinal results saved to {args.output}")
    else:
        print("No MATCH regions generated.")

if __name__ == "__main__":
    main()