# Omni-SV: A Knowledge-Guided Multi-Modal Genomic Foundation Model Framework for Structural Variant Filtering

**Omni-SV** is a pioneering knowledge-guided multi-modal framework that integrates **Genomic Foundation Models (GFMs)** into high-fidelity Structural Variant (SV) filtering. By synergizing syntactic structure features, deep evolutionary semantics, and functional genomic landscapes, Omni-SV transforms SV analysis from error-prone pattern matching to deep biological semantic understanding.

**Status:** Research Prototype (Based on ACMMM 2026 Conference Submission)

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

## Essential Bioinformatics Dependencies

| Package                                                      | Purpose                             |
| ------------------------------------------------------------ | ----------------------------------- |
| ![pysam](https://img.shields.io/badge/pysam-0.23.3-14a2b8?logo=python&logoColor=white) | BAM/CRAM file processing            |
| ![biopython](https://img.shields.io/badge/biopython-1.85-14a2b8?logo=python&logoColor=white) | Sequence analysis and manipulation  |
| ![mappy](https://img.shields.io/badge/mappy-2.30-14a2b8?logo=python&logoColor=white) | Sequence alignment and mapping      |
| ![cuteSV](https://img.shields.io/badge/cuteSV-2.1.3-14a2b8?logo=python&logoColor=white) | SV detection and calling            |
| ![sniffles](https://img.shields.io/badge/sniffles-2.6.3-14a2b8?logo=python&logoColor=white) | Long-read SV caller                 |
| ![intervaltree](https://img.shields.io/badge/intervaltree-3.1.0-14a2b8?logo=python&logoColor=white) | Genomic interval handling           |
| ![pyfaidx](https://img.shields.io/badge/pyfaidx-0.9.0.3-14a2b8?logo=python&logoColor=white) | FASTA file indexing and access      |
| ![PyVCF3](https://img.shields.io/badge/PyVCF3-1.0.4-14a2b8?logo=python&logoColor=white) | VCF file parsing and writing        |
| ![Truvari](https://img.shields.io/badge/Truvari-3.5.0-14a2b8?logo=python&logoColor=white) | SV benchmarking and comparison      |
| ![Badread](https://img.shields.io/badge/Badread-0.4.0-14a2b8?logo=python&logoColor=white) | Read simulation and quality control |
| ![SPOA](https://img.shields.io/badge/SPOA-4.1.5-14a2b8?logo=c%2B%2B&logoColor=white) | SIMD partial order alignment tool |
