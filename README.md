# MAID

## Datasets

Our experiments are based on the open-source datasets [DiffusionForensics](https://github.com/ZhendongWang6/DIRE), [Artifact](https://github.com/awsaf49/artifact) and [GenImage](https://github.com/Andrew-Zhu/GenImage).

## Installation

1. Clone this repository and navigate to the MAID folder:
   ```bash
   git clone https://github.com/Zhu-Luyu/MAID.git
   cd MAID
   ```

2. Install the required packages:
   ```bash
   conda env create -f environment.yaml -n maid
   conda activate maid
   ```

3. Download pre-trained diffusion models:

   - [DDIM](https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1)
   - The repository for Stable Diffusion v1.5 has been removed, but you can use [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) as a substitute.
   - [DiT](https://huggingface.co/facebook/DiT-XL-2-256)

   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   huggingface-cli download --resume-download CompVis/stable-diffusion-v1-4 --local-dir CompVis--stable-diffusion-v1-4
   ```

## Training & Evaluation

Modify the script parameters as needed to run training and evaluation:
   ```bash
   sh train.sh
   sh eval.sh
   ```

Modify the script parameters as needed to run DMA extraction:

   ```bash
   cd dma
   sh extract_dma.sh
   ```

## Acknowledgments

Our code is based on the frameworks provided by [CNNDetection](https://github.com/PeterWang512/CNNDetection) and [DNF](https://github.com/YichiCS/DNF). We greatly appreciate their contributions and code.
