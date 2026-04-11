#!/usr/bin/env python
# coding: utf-8

import os
import sys
import subprocess

def run_command(command):
    """Utility to run shell commands and handle errors."""
    print(f"[*] Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"[!] Error occurred while executing: {command}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    mode = sys.argv[1]

    # ==========================================================================
    # STAGE 1: PREPROCESS (Must be done first for either Training or Refinement)
    # ==========================================================================
    if mode == 'preprocess':
        # Usage: python OmniSV.py preprocess <vcf_path> <bam_path> <output_dir> <reference>
        if len(sys.argv) != 6:
            print("Usage: python OmniSV.py preprocess <vcf_path> <bam_path> <output_dir> <reference>")
            return
        
        vcf_path, bam_path, out_dir, ref_fa = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
        
        print("\n--- Step 1: Parsing VCF (parse_vcf.sh) ---")
        run_command(f"bash parse_vcf.sh {vcf_path} {out_dir}")

        print("\n--- Step 2: Data Preparation (data_main.py) ---")
        # Generates images and initial sequence reconstructions
        run_command(f"python data_prepare/data_main.py --bam {bam_path} --vcf {vcf_path} --ref {ref_fa} --out {out_dir}")

        print("\n--- Step 3: Feature Extraction for Training/Inference ---")
        # Before training, we must extract the triplet features (Evo2, AlphaGenome, Image)
        run_command(f"python feature_extract/evo2_encoder.py --data_dir {out_dir}")
        run_command(f"python feature_extract/alphagenome_encoder.py --data_dir {out_dir}")
        run_command(f"python feature_extract/image_encoder.py --data_dir {out_dir}")
        print(f"\n[+] Preprocessing complete. Data ready in {out_dir}")

    # ==========================================================================
    # STAGE 2: TRAIN (Uses data generated from PREPROCESS)
    # ==========================================================================
    elif mode == 'train':
        # Usage: python OmniSV.py train <label_json> <feat_dir> <save_dir> <model_name> <gpu_id>
        if len(sys.argv) != 7:
            print("Usage: python OmniSV.py train <label_json> <feat_dir> <save_dir> <model_name> <gpu_id>")
            return
        
        label_json, feat_dir, save_dir, model_name, gpu_id = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

        # Points to the three sub-folders generated in Preprocess
        syn_dir = os.path.join(feat_dir, "syn_feats")
        evo_dir = os.path.join(feat_dir, "evo_feats")
        omics_dir = os.path.join(feat_dir, "functional_feats")

        print("\n--- Starting Omni-SV Model Training (5-Fold CV) ---")
        # Training with F1-score optimization and MGO loss
        cmd = (f"python model_trainer.py "
               f"--label_json {label_json} --syn_dir {syn_dir} --evo_dir {evo_dir} --omics_dir {omics_dir} "
               f"--save_dir {save_dir} --model_name {model_name} --device cuda:{gpu_id} "
               f"--epochs 50 --batch_size 64")
        run_command(cmd)

    # ==========================================================================
    # STAGE 3: REFINE (Final Inference and VCF Revision)
    # ==========================================================================
    elif mode == 'refine':
        if len(sys.argv) != 7:
            print("Usage: python OmniSV.py refine <feat_dir> <model_path> <vcf_to_revise> <output_vcf> <gpu_id>")
            return

        feat_dir, model_path, raw_vcf, out_vcf, gpu_id = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        
        print("\n--- Running Model Prediction ---")
        temp_csv = os.path.join(feat_dir, "predictions.csv")
        run_command(f"python model_predictor.py --model_path {model_path} --data_dir {feat_dir} --output {temp_csv} --gpu {gpu_id}")

        print("\n--- Revising VCF based on Predictions ---")
        # Corrects DEL/INS and filters MATCH (False Positives)
        run_command(f"python revise_vcf.py --vcf {raw_vcf} --csv {temp_csv} --out {out_vcf}")

    else:
        print_usage()

def print_usage():
    print("\n" + "="*70)
    print("Omni-SV: Multi-modal Genomic SV Refiner Official Pipeline")
    print("="*70)
    print("\n[STEP 1] Preprocess (Run this first):")
    print("    python OmniSV.py preprocess <vcf> <bam> <out_dir> <ref.fa>")
    print("\n[STEP 2] Train (After Step 1, if building a new model):")
    print("    python OmniSV.py train <labels.json> <out_dir> <save_dir> <model_name> <gpu_id>")
    print("\n[STEP 3] Refine (After Step 1, using a trained model):")
    print("    python OmniSV.py refine <out_dir> <model_path> <vcf> <out_vcf> <gpu_id>")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()