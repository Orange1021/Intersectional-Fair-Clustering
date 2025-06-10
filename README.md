# Intersectional Fair Clustering: An Information-Theoretic Formulation

[![Paper-Link](https://img.shields.io/badge/Paper-Coming%20Soon-blue)](https://arxiv.org/abs/your_paper_id) <!-- Replace with your paper link once available -->
[![Conference](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/Conferences/AAAI-26/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official PyTorch implementation for our paper: [Intersectional Fair Clustering: An Information-Theoretic Formulation](https://arxiv.org/abs/your_paper_id)** (Paper under review)  
*Authors: Qincheng Zhao, Peng Zhang*

This repository provides the code for our novel framework that addresses intersectional fairness in clustering tasks. Our approach is grounded in a rigorous information-theoretic formulation that reveals the limitations of traditional fairness constraints and provides a principled way to mitigate bias arising from the intersection of multiple sensitive attributes.

This codebase is built upon the foundation of the CVPR 2023 paper **[FCMI](https://github.com/PengxinZeng/2023-CVPR-FCMI)**. We extend our sincere gratitude to the original authors for their excellent work.

---

## Core Insight: The Theory of Intersectional Fairness

Traditional fair clustering methods typically impose constraints on individual sensitive attributes (e.g., gender, race) independently. However, we theoretically demonstrate that this "additive" fairness assumption is insufficient, as it overlooks the **attribute crosstalk**—the interaction effects between attributes.

Our core theoretical finding, grounded in information theory, proves that the mutual information between the combined sensitive attributes and the clustering assignment is **always less than or equal to** the sum of the mutual information of individual attributes:

\\[ I(G_1, \\dots, G_S; C) \\leq \\sum_{s=1}^S I(G_s; C) \\]

**Key Implications**:
*   The equality holds if, and only if, all sensitive attributes are **conditionally independent** given the clustering assignment `C`.
*   In real-world scenarios, attributes are often correlated (e.g., a specific combination of race and gender may face unique biases), causing the strict inequality to hold.
*   Consequently, constraining individual attributes separately systematically underestimates the compounded bias faced by intersectional groups (e.g., "African-American women"). **Explicitly modeling the joint distribution of attributes is therefore a necessity** for achieving true intersectional fairness.

## Framework

To realize our theoretical goal, we propose a "Multi-Expert + Attention Fusion + Hierarchical Discriminators" framework:

1.  **Multi-Expert Network**: Each expert network independently learns representations for a single sensitive attribute to disentangle its marginal impact.
2.  **Attention Fusion**: An attention mechanism is used to dynamically learn the importance of different attribute combinations, generating a fused representation that captures intersectional effects.
3.  **Hierarchical Discriminators**:
    *   **Individual Discriminators**: Enforce fairness for each attribute separately.
    *   **Joint Discriminator**: A dedicated discriminator that operates on the joint distribution of attributes, directly enforcing our core theoretical constraint against intersectional bias.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Intersectional-Fair-Clustering.git
    cd Intersectional-Fair-Clustering
    ```

2.  **Create a Conda environment and install dependencies:**
    We recommend using Python 3.9 and PyTorch 1.11.0 as in the original project.
    ```bash
    conda create -n ifc python=3.9
    conda activate ifc
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    pip install -r requirements.txt
    
    # Install faiss for efficient clustering
    conda install -c pytorch faiss-gpu
    ```
    **`requirements.txt`:**
    ```
    numpy
    scikit-learn
    pandas
    tqdm
    ```

## Datasets

This project utilizes datasets commonly used in fair clustering research. Following the original FCMI implementation, we use:

*   **Office-31**: A standard domain adaptation dataset. We use the 'Amazon' and 'Webcam' domains, where the domain itself is a sensitive attribute.
*   **MNIST-USPS**: A combination of two digit recognition datasets, MNIST and USPS. The source dataset is treated as a sensitive attribute.
*   **ReverseMNIST**: A variant of MNIST where one group consists of original images and the other of color-inverted images.
*   **MTFL**: The Multi-Task Facial Landmark dataset, which includes attributes like wearing glasses and smiling that can be used as sensitive attributes.

We follow the setup from [DFC (Li et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Deep_Fair_Clustering_for_Visual_Learning_CVPR_2020_paper.pdf) to obtain these datasets.

Please download the datasets and place them in a `data/` directory. **Note**: The current data paths seem to be hard-coded in `DataSetMaster/dataset.py`. You may need to modify them to match your local directory structure.

## Usage

You can train and evaluate our model using `main.py`.

#### Training

Our framework is designed to handle multiple sensitive attributes and optimize for intersectional fairness.

```bash
# Train on MTFL with an intersectional fairness weight of 0.5.
python main.py --dataset MTFL --LambdaFair 0.5 --seed 9116

# Train on Office-31 with a higher fairness weight and over-sampling.
python main.py --dataset Office --LambdaFair 0.7 --sampling_method os --seed 0

# Train on MNIST-USPS
python main.py --dataset MNISTUSPS --LambdaFair 0.5 --seed 9116
```

**Key Arguments**:
*   `--dataset`: Specify the dataset (`MTFL`, `Office`, `MNISTUSPS`, `ReverseMNIST`).
*   `--LambdaFair`: **(Our Core Contribution)** The weight for the intersectional fairness loss term. This parameter controls the strength of the constraint on the **joint distribution** of multiple sensitive attributes. A value greater than 0 enables our proposed method.
*   `--LambdaClu`: The weight for the clustering-related loss terms.
*   `--sampling_method`: Optional data balancing method (`os`, `us`, `bm`).
*   `--seed`: The random seed for reproducibility.

#### Evaluation
To evaluate a trained model, use the `--resume` flag:
```bash
python main.py --dataset MTFL --seed 9116 --resume /path/to/your/model.pth --evaluate
```

## Key Arguments

We provide a range of command-line arguments to control the training process and model configuration. Here are some of the most important ones:

#### Core Fairness and Clustering Parameters

*   `--LambdaFair`: **(Our Core Contribution)** The weight for the intersectional fairness loss term. This parameter controls the strength of the constraint on the **joint distribution** of multiple sensitive attributes, corresponding to `β` in our paper. A value greater than 0 enables our proposed method. (Default: `0.20`)
*   `--LambdaClu`: The weight for the clustering-related loss terms (`InfoBalanceLoss` and `OneHot`). Corresponds to `α` in our paper. (Default: `0.04`)
*   `--Reconstruction`: The weight for the feature reconstruction loss. (Default: `1.0`)

#### Dataset and Sampling

*   `--dataset`: Selects the dataset to use. Choices: `MTFL`, `Office`, `MNISTUSPS`, `ReverseMNIST`.
*   `--sampling_method`: Data balancing method to apply to the training set. Choices: `os` (over-sampling), `us` (under-sampling), `bm` (bias-mimicking). (Default: `''`)

#### General Training

*   `--seed`: The random seed for reproducibility.
*   `--batch_size`: The training batch size. (Default: `512`)
*   `--train_epoch`: The total number of training epochs. (Default: `300`)
*   `--resume`: Path to a pre-trained model checkpoint to resume training or for evaluation.

## Citation

If you find our work or code useful in your research, please consider citing our paper:
```bibtex
@inproceedings{zhao2026intersectional,
  title   = {Intersectional Fair Clustering: An Information-Theoretic Formulation},
  author  = {Qincheng Zhao and Peng Zhang},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year    = {2026},
  url     = {https://arxiv.org/abs/your_paper_id}
}