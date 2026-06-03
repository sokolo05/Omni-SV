import pandas as pd
import pysam
import random
import argparse
import sys
import os

def calculate_large_indel_ratio(bam_file, chrom, start, end, min_indel_length=30):
    try:
        bam = pysam.AlignmentFile(bam_file, "rb")
    except Exception:
        return 1.0
        
    large_indel_reads = 0
    total_reads = 0
    
    try:
        for read in bam.fetch(chrom, start, end):
            total_reads += 1
            has_large_indel = False
            if read.cigartuples:
                for op, length in read.cigartuples:
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
    return not (end1 < start2 or start1 > end2)

def main():
    parser = argparse.ArgumentParser(description="Generate high-fidelity MATCH regions aligned with length distribution")
    
    parser.add_argument("-s", "--sv_file", required=True)
    parser.add_argument("-b", "--bam_file", required=True)
    parser.add_argument("-o", "--output", required=True)
    
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--buffer", type=int, default=3000)
    parser.add_argument("--max_ratio", type=float, default=0.1)
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.sv_file, sep="\t")
    except Exception as e:
        print(f"[-] CSV Read error: {e}")
        sys.exit(1)

    # Force absolute metrics synchronization
    if 'sv_len' in df.columns:
        df['sv_len'] = pd.to_numeric(df['sv_len'], errors='coerce').fillna(0).abs().astype(int)
    else:
        df['sv_len'] = (df['sv_end'] - df['sv_start']).abs()

    global_length_pool = [int(x) for x in df['sv_len'].tolist() if x > 0]
    if not global_length_pool:
        global_length_pool = [100] # Safe fallback anchor

    chrom_lengths = {}
    try:
        bam = pysam.AlignmentFile(args.bam_file, "rb")
        chrom_lengths = {str(ref): length for ref, length in zip(bam.references, bam.lengths)}
        bam.close()
    except Exception as e:
        print(f"[-] BAM Header read error: {e}")
        sys.exit(1)

    match_records = []
    
    for chrom in df["chr"].unique():
        chrom_str = str(chrom)
        if chrom_str not in chrom_lengths:
            print(f"[-] Warning: Chromosome {chrom_str} absent from BAM header reference layout.")
            continue
        
        chrom_len = chrom_lengths[chrom_str]
        current_chrom_svs = df[df["chr"] == chrom]
        
        total_sv = len(current_chrom_svs)
        num_target = int(total_sv * args.ratio)
        
        if num_target == 0:
            continue

        sv_exclusion_zones = []
        local_length_pool = [int(x) for x in current_chrom_svs['sv_len'].tolist() if x > 0]

        for _, row in current_chrom_svs.iterrows():
            e_start, e_end = int(row["sv_start"]), int(row["sv_end"])
            if row["sv_type"] == "INS":
                e_end = max(e_end, e_start + int(row["sv_len"]))
            sv_exclusion_zones.append((max(1, e_start - args.buffer), min(chrom_len, e_end + args.buffer)))
        
        sv_exclusion_zones.sort()

        gen_exclusion_zones = []
        generated = 0
        attempts = 0
        max_attempts = num_target * 300

        print(f"[*] Processing {chrom_str}: Generating {num_target} background MATCH constraints...")

        while generated < num_target and attempts < max_attempts:
            attempts += 1
            
            target_len = random.choice(local_length_pool if local_length_pool else global_length_pool)
            pos = random.randint(1, chrom_len)
            
            m_start = max(1, pos - (target_len // 2))
            m_end = min(chrom_len, m_start + target_len)
            
            if any(intervals_overlap(m_start, m_end, z[0], z[1]) for z in sv_exclusion_zones):
                continue
            
            if any(intervals_overlap(m_start, m_end, z[0], z[1]) for z in gen_exclusion_zones):
                continue

            ratio = calculate_large_indel_ratio(args.bam_file, chrom_str, m_start, m_end)
            if ratio >= args.max_ratio:
                continue

            match_id = f"MATCH_{chrom_str}_{generated:04d}"
            match_records.append([chrom_str, match_id, pos, "MATCH", target_len, m_start, m_end])
            
            gen_exclusion_zones.append((max(1, m_start - args.buffer), min(chrom_len, m_end + args.buffer)))
            generated += 1

        print(f"    -> Successfully spawned: {generated}/{num_target} controlled loci.")

    if match_records:
        df_match = pd.DataFrame(match_records, columns=["chr", "id", "pos", "sv_type", "sv_len", "sv_start", "sv_end"])
        pd.concat([df, df_match], ignore_index=True).to_csv(args.output, sep="\t", index=False)
        print(f"[+] Multi-modal negative mining finalized. Exported downstream file: {args.output}")
    else:
        print("[-] Execution halted: No validated MATCH instances harvested.")

if __name__ == "__main__":
    main()
