#!/bin/bash

# Usage: chmod 755 parse_vcf.sh
# Usage: ./parse_vcf.sh <input.vcf.gz> <output.txt>

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.vcf[.gz]> <output.txt>"
    echo "Example: $0 data.vcf.gz result.txt"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found!"
    exit 1
fi

case "$INPUT" in
    *.gz)
        READ_CMD="zcat"
        ;;
    *)
        READ_CMD="cat"
        ;;
esac

# Write standardized header matching pandas downstream consumption attributes
printf "chr\tid\tpos\tsv_type\tsv_len\tsv_start\tsv_end\n" > "$OUTPUT"

# Core Processing Logic with stabilized absolute variance length conversions
$READ_CMD "$INPUT" | awk -F '\t' 'BEGIN{OFS="\t"}
!/^#/ {
  chrom = $1;
  pos = $2;
  id = $3;
  info = $8;

  svtype = "NA";
  svlen = 0;

  n = split(info, arr, ";");
  for (i = 1; i <= n; i++) {
    split(arr[i], kv, "=");
    if (kv[1] == "SVTYPE") svtype = kv[2];
    if (kv[1] == "SVLEN") svlen = kv[2];
  }

  # Rigid standardization: enforce absolute values for svlen to eliminate character variance
  if (svlen < 0) {
    svlen = -svlen;
  }
  
  # Handle empty or zero lengths gracefully by string evaluation
  if (svlen == "" || svlen == 0) {
    svlen = "NA";
  }

  sv_start = pos;

  if (svtype == "INS") {
    sv_end = pos + 1;
  } else {
    if (svlen != "NA") {
      sv_end = pos + svlen;
    } else {
      sv_end = pos + 1;
    }
  }

  print chrom, id, pos, svtype, svlen, sv_start, sv_end;
}' >> "$OUTPUT"

echo "[+] VCF standardization complete. Exported to: $OUTPUT"
