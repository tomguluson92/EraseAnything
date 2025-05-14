# EraseAnything (ICML 2025)

<p align="center">
  <a href="https://arxiv.org/abs/2412.20413">
    <img src='https://img.shields.io/badge/Paper-arXiv%20Preprint-green?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
  </a>
  <a href='https://tomguluson92.github.io/projects/eraseanything/'>
    <img src='https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'>
  </a>
</p>

![Teaser](teaser.png)

EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers is an open-source project for **Concept Erasing in Rectified Flow models: e.g. Flux, SD 3**.


## Features

- ✅ Supports **[diffusers]**
- ✅ Easy to extend and integrate

## Installation & Environment Setup

1. **Install Rust (if required):**
   ```bash
   curl https://sh.rustup.rs -sSf | sh
   export PATH="$HOME/.cargo/bin:$PATH"
   ```

2. **Install Python dependencies:**
   ```bash
   pip install transformers sentencepiece einops omegaconf
   pip install tokenizers==0.20.0
   pip install nltk wandb openai
   ```

3. **Install local packages:**
   ```bash
   cd peft
   pip install -e .[torch]

   or

   python setup.py install

   cd ../diffusers
   pip install -e .[torch]

   or

   python setup.py install
   ```

## Quick Start

0. **Set AI Secret Key** (Strongly recommend OpenRouter!)

    ```bash
    API_KEY='xxx'
    END_POINT='https://research-01-02.openai.azure.com/'
    API_VERSION = "2024-08-01-preview"
    ```

1. **Train the model:**
   ```bash
   bash train.sh
   ```

2. **Run inference:**
   ```bash
   python inference.py
   ```

## Training & Inference

- Training script: `train.sh`
- Inference script: `inference.py`
- Results will be saved in the `results/` directory (if applicable).

## Acknowledgments

This project is inspired by and builds upon the work of [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing) and other open-source projects. We thank the community for their valuable contributions.

## Citation

If you use this project in your research, please cite:
```bibtex
@article{gao2024eraseanything,
  title={EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers},
  author={Gao, Daiheng and Lu, Shilin and Walters, Shaw and Zhou, Wenbo and Chu, Jiaming and Zhang, Jie and Zhang, Bang and Jia, Mengxi and Zhao, Jian and Fan, Zhaoxin and others},
  journal={ICML 2025},
  year={2024}
}
```

## Contact

For technical questions, please contact **samuel.gao023@gmail.com**.
