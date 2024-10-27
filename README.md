# VAE-Palette: Generative Model to Recover Missing Data from Collimated X-Ray Images


This repo contains the code for the master thesis **Optimizing X-Ray Imaging: Leveraging Generative Models to Recover Missing Data from Collimated Images**.


#### Author: Nguyen Toan, LE - Technical University of Munich

#### Advisor: Mohammad Farid Azampour, Agnieszka Tomczak

## Contents

1. [Abstract](#Abstract)
2. [Metrics](#metrics)
    - [MSE Score](#mse-score)
    - [SSIM Score](#ssim-score)
    - [LPIPS Score](#lpips-score)
3. [Relevant Repositories](#relevant-repositories)
4. [Installation Instructions](#installation-instructions)
5. [Execution Instructions](#execution-instructions)
    - [Training Stratergies](#training-stratergies)
    - [Inference Stratergies](#inference-stratergies)
    - [First Training Phase](#first-training-phase---data-preprocessing)
    - [Second Training Phase](#second-training-phase---training-palette)
    - [Testing](#testing---inference)
    - [Evaluation](#evaluation)
6. [Results](#results)
7. [Citation](#citation)

## Abstract

X-ray imaging provides surgeons with a static roadmap for anatomical reference and tool navigation during cardiovascular procedures but exposes patients to high radiation doses. This radiation dose is reduced by collimating the X-ray beam, but this causes limited anatomical context for clinicians. To overcome this, a method to reconstruct full-field-of-view images is proposed using the collimated data intra-operatively. This approach combines a Variational Autoencoder with Palette, a diffusion model, to enhance image reconstruction performance. This results in an improvement of 29.97% in MSE, 4.48% in SSIM, and 62.08% in LPIPS compared to the diffusion model alone. This method outperforms pix2pixHD in resolution while maintaining comparable metrics. This work aims to make collimators more commonly used in clinical routines to diminish radiation exposure without decreasing the quality of visual information.

## Metrics

### MSE Score

- This score measures the pixel-wise difference between two images. It calculates the average of the squared errors, which represents the average squared difference between the calculated and actual values.
- A lower MSE value shows that the two images are more similar.

### SSIM Score

- This score measures the structural similarity between two images. It evaluates differences in luminance, contrast, and structural information.
- SSIM ranges between −1 and 1, where 1 indicates perfect structural similarity. 
- A higher SSIM score indicates that the two images are more similar.

### LPIPS Score

- This score is a perceptual metric based on deep learning. It calculates the perceptual similarity between images using features extracted
from a pre-trained deep neural network, in this case, the encoder part of Models Genesis ([Zhou et al., 2021](#citation)). 
- LPIPS score has the advantage of capturing the perceptual differences that align more closely with human visual perception.
- The LPIPS metric passes the original and generated images through a deep neural network to extract features at different layers. Then, it is computed as the L1 distance between the feature representations of the image pairs. 
- A lower LPIPS score indicates that the two images are more perceptually similar.

## Relevant Repositories

- Back-bone: [Uncropping task of Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
- Pre-processor: [Vanilla VAE](https://github.com/AntixK/PyTorch-VAE/tree/master) 

## Installation Instructions

- I use Python 3.10.12, PyTorch 2.4.0 (CUDA 11.0 build).
- I trained the model on an NVIDIA Quadro RTX 6000 GPU with 24GB of memory.
- To install environments, use this command:
```python
pip install -r requirements.txt
```

## Preparation

You can change any hyperparameters and some other relevant parameters (data path, path to pre-trained weights, saving path etc.) by modifying these files:
- VAE: [vae.yaml](./VAE/configs/vae.yaml)
- Training Palette: [uncropping_custom_train.json](./Palette/config/uncropping_custom_train.json)
- Testing Palette: [uncropping_custom_test.json](./Palette/config/uncropping_custom_test.json)
- Shell script command: [Training Palette](./run_train_Palette.sh), [Evaluating Palette](./run_eval_Palette.sh), and [Testing Palette](./run_test_Palette.sh)

In the shell scripts, please change the environment parameters (MINICONDA_DIR, CONDA_ENV)

In these files, I used the absolute paths for all path parameters. You can adapt these variables (in config scripts and in shell script) to fit to your project.

## Execution Instructions

### Training Stratergies

<img src="images/Training.png" width="30%"/>

### Inference Stratergies

<img src="images/Inference.png" width="30%"/>

### First Training Phase - Data Preprocessing

```python
cd VAE
python run.py
```

### Second Training Phase - Training Palette

```python
cd ..
cd Palette
./run_train_Palette.sh
```

### Testing - Inference

```python
./run_test_Palette.sh
```

### Evaluation

```python
./run_eval_Palette.sh
```
This script returns:
- LPIPS scores of each Collimated-GT image pair
- Avarage and STD of LPIPS score of all images in test set
- Avarage and STD of MSE score of all images in test set
- Avarage and STD of SSIM score of all images in test set

## Results

- Statistically, the evaluation scores of the test set during inference are summarized in this table:

| MSE (&darr;) | SSIM (&uarr;) | LPIPS (e-05) (&darr;) |
| :---:| :---: |  :---: |
| 0.071698 &plusmn; 0.018560 | 0.217399 &plusmn; 0.086350 | 5.308796 &plusmn; 3.089877 |

- Visually, the results are divided into three groups
    - Good generation:

    <img src="images/Good_results.png" width="30%"/>

    - Generation with incomplete information:

    <img src="images/Undetailed_results.png" width="30%"/>

    - Bad generation:

    <img src="images/Bad_results.png" width="30%"/>

## Citation

If you found VAE-Palette useful in your research, please consider starring ⭐ me on GitHub and citing 📚 us in your research!

```bibtex
@article{Zhou_2021,
    title={Models Genesis},
    volume={67},
    ISSN={1361-8415},
    url={http://dx.doi.org/10.1016/j.media.2020.101840},
    DOI={10.1016/j.media.2020.101840},
    journal={Medical Image Analysis},
    publisher={Elsevier BV},
    author={Zhou, Zongwei and Sodha, Vatsal and Pang, Jiaxuan and Gotway, Michael B. and Liang, Jianming},
    year={2021},
    month=jan, pages={101840} 
    }
```

## Acknowledgement

I would like to thank the authors of [OneFormer](https://github.com/SHI-Labs/OneFormer/tree/main), [Palette](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models), and [VAE](https://github.com/AntixK/PyTorch-VAE/tree/master) for releasing their helpful codebases.