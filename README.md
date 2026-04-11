# Omni-SV: A Knowledge-Guided Multi-Modal Genomic Foundation Model Framework for Structural Variant Filtering

**Omni-SV** is a pioneering knowledge-guided multi-modal framework that integrates **Genomic Foundation Models (GFMs)** into high-fidelity Structural Variant (SV) filtering. By synergizing syntactic structure features, deep evolutionary semantics, and functional genomic landscapes, Omni-SV transforms SV analysis from error-prone pattern matching to deep biological semantic understanding.

**Status:** Research Prototype (Based on ACMMM 2026 Conference Anonymous Submission Id: 9174)

---

## 📖 Introduction

Detecting structural variations (SVs) from long-read sequencing data remains a critical challenge due to high stochastic noise and complex genomic contexts. Traditional methods often fail to capture deep biological semantics, relying primarily on raw alignment signals.

Omni-SV addresses this by introducing a **trinitarian synergistic perception framework**:
1.  **Syntactic Structure Modality:** Extracts alignment trajectories (CIGAR strings).
2.  **Evolutionary Semantic Modality:** Leverages **Evo2** to generate Shannon entropy landscapes and spatiotemporal attention maps.
3.  **Functional Genomic Modality:** Utilizes **AlphaGenome** to project candidates onto high-resolution epigenetic landscapes.

We introduce the **Evolutionary-Anchored Asymmetric Neural Rectification (EAANR)** mechanism, which uses evolutionary priors to dynamically audit and rectify noisy syntactic signals. This ensures robust decision-making even when physical signals are ambiguous.

### Visual Representation of Modalities
Below is the overview of the Multi-modal Co-perception Framework as described in our research:

<!-- Insert Figure 1 from Paper Here -->
<p align="center">
  <img src="images/01.Multimodal-2.png" alt="Omni-SV: Multi-modal Co-perception Framework" width="800">
  <br>
  <em>Figure 1: The Multi-modal Co-perception Framework. Omni-SV synergizes syntactic structures, evolutionary semantics (Evo2), and functional genomics (AlphaGenome) for SV representation.</em>
</p>

---

## 🏗️ Model Architecture

The Omni-SV architecture is designed to bridge the gap between noisy genomic signals and high-level biological priors through a hierarchical integration paradigm.

The pipeline consists of three stages:
1.  **Heterogeneous Modality Representation:** Projects inputs into a unified latent space.
2.  **EAANR Module:** Performs asymmetric rectification using evolutionary constraints.
3.  **Multi-omics Functional Landscape Modulation:** Calibrates features against the biochemical background.

### Architecture Diagram
The figure below illustrates the detailed workflow of data preprocessing, feature representation, and optimization:

<!-- Insert Figure 2 from Paper Here -->
<p align="center">
  <img src="images/02.main_plot-2.png" alt="Omni-SV: Methodology Overview" width="800">
  <br>
  <em>Figure 2: The overall architecture of Omni-SV. (a) Data preprocessing; (b) Heterogeneous modality representation; (c) EAANR; (d) Functional Landscape Modulation; (e) Optimization.</em>
</p>

---

## 📊 Key Features

*   **🧬 Knowledge-Guided Fusion:** Unlike traditional methods, we use pre-trained Genomic Foundation Models (Evo2 and AlphaGenome) as prior knowledge to guide the filtering process.
*   **🛡️ EAANR Mechanism:** Our proprietary Evolutionary-Anchored Asymmetric Neural Rectification module uses entropy-gating to suppress false positives in unstable genomic regions.
*   **🌍 Cross-Species Generalization:** Validated on both *Homo sapiens* (Human) and *Arabidopsis thaliana* genomes.
*   **⚡ Multi-Granularity Optimization:** A joint loss function that ensures representation consistency and mutual information maximization.

---

## 🚀 Getting Started

This section outlines the basic setup required to run Omni-SV based on the reference implementation.

### Prerequisites

*   **Python:** >= 3.11
*   **PyTorch:** 2.1
*   **Hardware:** NVIDIA GPU (Recommended: 24GB显存 RTX 4090 6152)
*   **Dependencies:** pandas, numpy, biopython, pysam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sokolo05/Omni-SV.git
cd Omni-SV

# 2. Create a virtual environment (recommended)
python -m venv omni-sv-env
source omni-sv-env/bin/activate # On Windows: omni-sv-env\Scripts\activate

# 3. Install PyTorch (Select the appropriate command for your CUDA version)
# Example for CUDA 12.4:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Install other requirements
pip install -r requirements.txt
```

## 📚 Essential Bioinformatics Dependencies

| Dependency & Logo | Description |
| :--- | :--- |
| [![SPOA](https://img.shields.io/badge/SPOA-4.1.5-00599C?logo=c%2B%2B&logoColor=white)](https://github.com/rvaser/spoa) | SIMD partial order alignment for noise reduction |
| [![EVO2](https://img.shields.io/badge/EVO2-v2.0-blue?logo=google-cloud&logoColor=white)](https://github.com/ArcInstitute/evo2) | GFM for evolutionary sequence embeddings |
| [![AlphaGenome](https://img.shields.io/badge/AlphaGenome-v1.0-blue?logo=google-cloud&logoColor=white)](https://github.com/google-deepmind/alphagenome) | Genomic foundation model for functional semantics |
| [![pysam](https://img.shields.io/badge/pysam-0.23.3-14a2b8?logo=python&logoColor=white)](https://github.com/pysam-developers/pysam) | BAM/CRAM file processing |
| [![biopython](https://img.shields.io/badge/biopython-1.85-14a2b8?logo=python&logoColor=white)](https://github.com/biopython/biopython) | Sequence analysis and manipulation |
| [![cuteSV](https://img.shields.io/badge/cuteSV-2.1.3-14a2b8?logo=python&logoColor=white)](https://github.com/tjiangHIT/cuteSV) | Signature-based long-read SV caller |
| [![sniffles](https://img.shields.io/badge/sniffles-2.6.3-00599C?logo=c%2B%2B&logoColor=white)](https://github.com/fritzsedlazeck/Sniffles) | High-throughput long-read SV caller |
| [![SVIM](https://img.shields.io/badge/SVIM-2.0.0-14a2b8?logo=python&logoColor=white)](https://github.com/eldariont/svim) | SV identification using long-read mappings |
| [![CSV-filter](https://img.shields.io/badge/CSV--filter-v1.0-blue?logo=github&logoColor=white)](https://github.com/xcxw127/CSV-Filter) | Collaborative SV filtering baseline |
| [![MMF-SV](https://img.shields.io/badge/MMF--SV-v1.0-blue?logo=pytorch&logoColor=white)](https://github.com/xcxw127/MMF-SV) | Multi-modal fusion for structural variation |
| [![PyVCF3](https://img.shields.io/badge/PyVCF3-1.0.4-14a2b8?logo=python&logoColor=white)](https://github.com/jdoughertyii/PyVCF) | VCF file parsing and writing |
| [![Truvari](https://img.shields.io/badge/Truvari-3.5.0-14a2b8?logo=python&logoColor=white)](https://github.com/ACEnglish/truvari) | SV benchmarking and comparison |
| [![SURVIVOR](https://img.shields.io/badge/SURVIVOR-1.0.7-00599C?logo=c%2B%2B&logoColor=white)](https://github.com/fritzsedlazeck/SURVIVOR) | Tool for merging and comparing SV calls |

## 🛠️ Data Preparation (Preprocessing)

The Omni-SV pipeline requires a specific preprocessing stage to generate the necessary triplet features (Syntactic, Evolutionary, Functional) from raw sequencing data. This is handled by the `preprocess` mode in `OmniSV.py`.

### Preprocessing Command

You must run this step first to prepare the data before training or refinement.

```
python OmniSV.py preprocess \
    <vcf_path> \
    <bam_path> \
    <output_dir> \
    <reference_fasta>
```

### Arguments

- `vcf_path`: Path to the input VCF file containing candidate SVs.
- `bam_path`: Path to the input BAM alignment file.
- `output_dir`: Directory where processed data and features will be saved.
- `reference_fasta`: Path to the reference genome (`.fa`).

### Pipeline Steps

The `preprocess` mode executes the following sub-tasks sequentially:

1. **Parse VCF:** Runs `parse_vcf.sh` to standardize the VCF input.
2. **Data Preparation:** Executes `data_main.py` to generate images and initial sequence reconstructions.
3. **Feature Extraction:**
    - **Evo2:** Runs `evo2_encoder.py` to extract evolutionary features.
    - **AlphaGenome:** Runs `alphagenome_encoder.py` to extract functional genomic features.
    - **Image:** Runs `image_encoder.py` to process syntactic images.

## 🚀 Model Training

Once the data is preprocessed, you can train the Omni-SV model using the `train` mode. This stage utilizes the extracted features to perform 5-Fold Cross-Validation with F1-score optimization.

### Training Command

```
python OmniSV.py train \
    <label_json> \
    <feat_dir> \
    <save_dir> \
    <model_name> \
    <gpu_id>
```

### Arguments

- `label_json`: Path to the JSON file containing ground truth labels (generated during preprocessing).
- `feat_dir`: The same `<output_dir>` used in the preprocessing step. The script automatically looks for `syn_feats`, `evo_feats`, and `functional_feats` subfolders inside it.
- `save_dir`: Directory where the trained model checkpoints will be saved.
- `model_name`: Name of the model to be saved.
- `gpu_id`: GPU device ID to use (e.g., `0`).

### Training Details

- **Optimizer:** AdamW with F1-score based checkpointing.
- **Loss Function:** Multi-Granularity Optimization (MGO) loss.
- **Hyperparameters:** Default epochs set to 50, batch size 64 (defined in `model_trainer.py`).

## 🧪 Result Refinement

The `refine` mode performs inference on new data and revises the original VCF file. It filters out false positives (MATCH) and corrects SV types (DEL/INS) based on model predictions.

### Refinement Command

```
python OmniSV.py refine \
    <feat_dir> \
    <model_path> \
    <vcf_to_revise> \
    <output_vcf> \
    <gpu_id>
```

### Arguments

- `feat_dir`: Directory containing the extracted features for inference.
- `model_path`: Path to the trained model checkpoint (`.pth`).
- `vcf_to_revise`: Path to the original VCF file you wish to improve.
- `output_vcf`: Path where the refined, high-fidelity VCF file will be saved.
- `gpu_id`: GPU device ID to use.

### Workflow

1. **Prediction:** Runs `model_predictor.py` to generate predictions (saved as `predictions.csv`).
2. **Revision:** Runs `revise_vcf.py` to update the VCF file based on the predictions, filtering false positives and correcting variant types.

## 📦 Full Pipeline Example

### Complete End-to-End Workflow

```
# Step 1: Preprocessing
python OmniSV.py preprocess \
    ./data/raw_calls.vcf \
    ./data/alignments.bam \
    ./processed_data/ \
    ./reference/hg38.fa

# Step 2: Training (optional if using pre-trained model)
python OmniSV.py train \
    ./processed_data/labels.json \
    ./processed_data/ \
    ./models/ \
    omni_sv_model \
    0

# Step 3: Refinement
python OmniSV.py refine \
    ./processed_data/ \
    ./models/omni_sv_model.pth \
    ./data/raw_calls.vcf \
    ./results/refined_calls.vcf \
    0
```

## 📚 Documentation

For detailed documentation on each component, please refer to the individual module documentation:

- `parse_vcf.sh`: VCF parsing utility
- `data_prepare/`: Data preprocessing modules
- `feature_extract/`: Feature extraction pipelines
- `model_trainer.py`: Training implementation
- `model_predictor.py`: Inference implementation
- `revise_vcf.py`: VCF refinement utility

## 🙏 Acknowledgments

- The developers of Evo2 and AlphaGenome models
- The open-source bioinformatics community
- The PyTorch development team
