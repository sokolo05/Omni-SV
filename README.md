Omni-SV: A Knowledge-Guided Multi-Modal Genomic Foundation Model Framework for Structural Variant Filtering
[





](LICENSE)
Omni-SV is a pioneering knowledge-guided multi-modal framework that integrates Genomic Foundation Models (GFMs) into high-fidelity Structural Variant (SV) filtering. By synergizing syntactic structure features, deep evolutionary semantics, and functional genomic landscapes, Omni-SV transforms SV analysis from error-prone pattern matching to deep biological semantic understanding.
Status: Research Prototype (Based on Conference Submission)
Current Date: April 11, 2026



📖 Introduction
Detecting structural variations (SVs) from long-read sequencing data remains a critical challenge due to high stochastic noise and complex genomic contexts. Traditional methods often fail to capture deep biological semantics, relying primarily on raw alignment signals.
Omni-SV addresses this by introducing a trinitarian synergistic perception framework:
1. Syntactic Structure Modality: Extracts alignment trajectories (CIGAR strings).
2. Evolutionary Semantic Modality: Leverages Evo2 to generate Shannon entropy landscapes and spatiotemporal attention maps.
3. Functional Genomic Modality: Utilizes AlphaGenome to project candidates onto high-resolution epigenetic landscapes.
We introduce the Evolutionary-Anchored Asymmetric Neural Rectification (EAANR) mechanism, which uses evolutionary priors to dynamically audit and rectify noisy syntactic signals. This ensures robust decision-making even when physical signals are ambiguous.
Visual Representation of Modalities
Below is the overview of the Multi-modal Co-perception Framework as described in our research:



🏗️ Model Architecture
The Omni-SV architecture is designed to bridge the gap between noisy genomic signals and high-level biological priors through a hierarchical integration paradigm.
The pipeline consists of three stages:
1. Heterogeneous Modality Representation: Projects inputs into a unified latent space.
2. EAANR Module: Performs asymmetric rectification using evolutionary constraints.
3. Multi-omics Functional Landscape Modulation: Calibrates features against the biochemical background.
Architecture Diagram
The figure below illustrates the detailed workflow of data preprocessing, feature representation, and optimization:



📊 Key Features
● 🧬 Knowledge-Guided Fusion: Unlike traditional methods, we use pre-trained Genomic Foundation Models (Evo2 and AlphaGenome) as prior knowledge to guide the filtering process.
● 🛡️ EAANR Mechanism: Our proprietary Evolutionary-Anchored Asymmetric Neural Rectification module uses entropy-gating to suppress false positives in unstable genomic regions.
● 🌍 Cross-Species Generalization: Validated on both Homo sapiens (Human) and Arabidopsis thaliana genomes.
● ⚡ Multi-Granularity Optimization: A joint loss function that ensures representation consistency and mutual information maximization.



🚀 Getting Started
This section outlines the basic setup required to run Omni-SV based on the reference implementation.
Prerequisites
● Python: >= 3.9
● PyTorch: 2.1
● Hardware: NVIDIA GPU (Recommended: RTX 4090 / A100 with 24GB+ VRAM)
● Dependencies: pandas, numpy, biopython, pysam
Installation
代码
图标/24_new/复制

# 1. Clone the repository
git clone https://github.com/sokolo05/Omni-SV.git
cd Omni-SV

# 2. Create a virtual environment (recommended)
python -m venv omni-sv-env
source omni-sv-env/bin/activate # On Windows: omni-sv-env\Scripts\activate

# 3. Install PyTorch (Select the appropriate command for your CUDA version)
# Example for CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Install other requirements
pip install -r requirements.txt

Data Preparation
Omni-SV requires three input modalities. Please refer to the data/ directory for expected formats.
1. Syntactic Structure (CIGAR Images): Generated from BAM alignment files.
2. Nucleotide Consensus Sequence: Generated using Partial Order Alignment (POA).
3. Multi-omics Functional Tracks: 14-track functional signals compatible with AlphaGenome.
See scripts/preprocess.sh for example preprocessing commands using minimap2 and samtools.



🧪 Usage
Training
To train the model on a custom dataset, use the following command:
代码
图标/24_new/复制

python train.py \
    --syntactic_data_path /path/to/cigar/images \
    --sequence_data_path /path/to/consensus/sequences \
    --functional_data_path /path/to/functional/tracks \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4

Evaluation
To evaluate a pre-trained model:
代码
图标/24_new/复制

python evaluate.py --checkpoint_path ./checkpoints/best_model.pth --test_data /path/to/test/set




📈 Results
Omni-SV establishes a new state-of-the-art (SOTA) in SV filtering across multiple platforms:
Platform
Metric (F1-Score)
Improvement vs. Baseline
PacBio CLR
0.9218
+1.49% vs. MMF-SV
Oxford Nanopore (ONT)
0.9356
+0.91% vs. MMF-SV


The framework demonstrates robustness across species, achieving high accuracy on both Human and A. thaliana genomes.



📚 Citation
If you use this code or the concepts described herein in your research, please cite our work:
代码
图标/24_new/复制

@misc{Anonymous2026OmniSV,
  title        = {Omni-SV: A Knowledge-Guided Multi-Modal Genomic Foundation Model Framework for Structural Variant Filtering},
  author       = {Anonymous Author(s)},
  year         = 2026,
  booktitle    = {Conference Acronym 'XX},
  submission_id = {9174}
}

Note: Please replace the anonymous details with the final publication information upon acceptance.



🛡️ License
This project is licensed under the MIT License - see the  LICENSE  file for details.



🧑‍💻 Acknowledgements
● Evo2 and AlphaGenome models are adapted from their respective foundational works.
● We thank the developers of PyTorch and Hugging Face for providing essential deep learning infrastructure.
