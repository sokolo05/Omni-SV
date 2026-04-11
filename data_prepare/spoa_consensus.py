import subprocess
import tempfile
import os
from typing import List

class SpoaEngine:
    def __init__(self, match=5, mismatch=-4, gap_open=-8, gap_extend=-6):
        """
        Store alignment parameters. 
        Note: Local spoa uses -m, -n, -g, -e flags.
        """
        self.match = str(match)
        self.mismatch = str(mismatch)
        self.gap_open = str(gap_open)
        self.gap_extend = str(gap_extend)

    def generate_consensus(self, sequences: List[str]) -> str:
        """
        Runs the local spoa binary via a temporary FASTA file.
        """
        # Remove empty sequences
        sequences = [s for s in sequences if s.strip()]
        if not sequences:
            return ""

        # Create a temporary FASTA file for spoa input
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_in:
            for i, seq in enumerate(sequences):
                tmp_in.write(f">seq_{i}\n{seq}\n")
            tmp_input_path = tmp_in.name

        try:
            # Build the command
            # -l 1: Global alignment
            # -r 0: Output consensus only
            cmd = [
                "spoa", 
                "-m", self.match,
                "-n", self.mismatch,
                "-g", self.gap_open,
                "-e", self.gap_extend,
                "-l", "1", 
                "-r", "0",
                tmp_input_path
            ]

            # Execute spoa
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # The output of spoa -r 0 is a FASTA format string:
            # >Consensus
            # ACTG...
            lines = process.stdout.strip().split('\n')
            
            if len(lines) >= 2:
                # Return only the sequence part (skip the header line)
                return "".join(lines[1:]).strip()
            return ""

        except subprocess.CalledProcessError as e:
            print(f"[SPOA CLI Error] Command failed: {e.stderr}")
            return ""
        except Exception as e:
            print(f"[SPOA CLI Error] Unexpected error: {e}")
            return ""
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)