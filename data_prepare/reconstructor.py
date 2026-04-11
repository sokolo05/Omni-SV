import pysam
from typing import List, Dict

class GenomicReconstructor:
    def __init__(self, bam_path: str, ref_path: str):
        self.bam = pysam.AlignmentFile(bam_path, "rb")
        self.ref = pysam.FastaFile(ref_path)

    def reconstruct_full_reads(self, chrom: str, pos: int, extend_len: int, max_reads: int = 30) -> List[Dict]:
        target_0based = pos - 1
        start, end = max(0, target_0based - extend_len), target_0based + extend_len
        
        extracted_data = []
        for read in self.bam.fetch(chrom, start, end):
            if read.is_unmapped or not read.cigartuples:
                continue

            q_seq = read.query_sequence
            curr_ref_pos = read.reference_start
            curr_q_pos = 0
            
            full_cigar, full_base, ref_mapping = [], [], {}

            for op, length in read.cigartuples:
                op_char = "MIDNSHP=X"[op]
                for _ in range(length):
                    if op in (0, 7, 8): # M, =, X
                        ref_mapping[curr_ref_pos] = len(full_cigar)
                        full_cigar.append(op_char)
                        full_base.append(q_seq[curr_q_pos] if q_seq else "N")
                        curr_q_pos += 1
                        curr_ref_pos += 1
                    elif op == 1: # I
                        full_cigar.append('I')
                        full_base.append(q_seq[curr_q_pos] if q_seq else "N")
                        curr_q_pos += 1
                    elif op == 2: # D
                        ref_mapping[curr_ref_pos] = len(full_cigar)
                        full_cigar.append('D')
                        try:
                            ref_b = self.ref.fetch(chrom, curr_ref_pos, curr_ref_pos+1).upper()
                        except:
                            ref_b = "N"
                        full_base.append(ref_b)
                        curr_ref_pos += 1
                    elif op == 4: # S
                        full_cigar.append('S')
                        full_base.append(q_seq[curr_q_pos] if q_seq else "N")
                        curr_q_pos += 1

            if target_0based in ref_mapping:
                extracted_data.append({
                    'cigar': full_cigar,
                    'base': full_base,
                    'center_idx': ref_mapping[target_0based],
                    'mapq': read.mapping_quality
                })

        extracted_data.sort(key=lambda x: x['mapq'], reverse=True)
        return extracted_data[:max_reads]

    def close(self):
        self.bam.close()
        self.ref.close()