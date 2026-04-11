#!/bin/bash

# Usage: chmod 755 parse_vcf.sh
# Usage: ./parse_vcf.sh <input.vcf.gz> <output.txt>

# 1. Parameter Check
# Check if at least 2 arguments (input and output files) are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.vcf[.gz]> <output.txt>"
    echo "Example: $0 data.vcf.gz result.txt"
    exit 1
fi

# 2. Get Parameters
INPUT="$1"
OUTPUT="$2"

# 3. Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found!"
    exit 1
fi

# 4. Determine how to read the file based on extension
# Using case statement for better compatibility with /bin/sh
case "$INPUT" in
    *.gz)
        READ_CMD="zcat"
        ;;
    *)
        READ_CMD="cat"
        ;;
esac

# 5. Write Header
# Use  to ensure \t is interpreted as a tab character
printf "chr\tid\tpos\tsv_type\tsv_len\tsv_start\tsv_end\n" > "$OUTPUT"
# 6. Core Processing Logic
# Use the determined command ($READ_CMD) to read the file and awk for parsing
$READ_CMD "$INPUT" | awk -F '\t' 'BEGIN{OFS="\t"}
!/^#/ {
  # Extract basic fields
  chrom = $1;
  pos = $2;
  id = $3;
  info = $8;

  # Initialize variables
  svtype = ".";
  svlen = ".";

  # Parse INFO field
  n = split(info, arr, ";");
  for (i = 1; i <= n; i++) {
    split(arr[i], kv, "=");
    if (kv[1] == "SVTYPE") svtype = kv[2];
    if (kv[1] == "SVLEN") svlen = kv[2];
  }

  # Handle negative SVLEN (take absolute value)
  if (svlen < 0) svlen = -svlen;

  # Calculate start and end positions
  sv_start = pos;

  if (svtype == "INS") {
    sv_end = pos + 1;
  } else {
    sv_end = pos + svlen;
  }

  # Output result
  print chrom, id, pos, svtype, svlen, sv_start, sv_end;
}' >> "$OUTPUT"

echo "Processing complete. Output saved to: $OUTPUT"
