# Sentinel-2 Water Segmentation

This repository contains the code, model definitions, configuration files, and dataset preparation utilities used for water-body extraction from Sentinel-2 multispectral imagery. The project includes single-model training and inference pipelines, as well as ensemble-based prediction strategies.

The repository is intended to make the experimental pipeline reproducible. The full research dataset is not distributed in this repository.

## 1. Scope and Main Functions

This project supports the following typical use cases:

- train a water-segmentation model from Sentinel-2 image tiles and binary masks
- evaluate a trained model on a validation or test split
- run water-body extraction on new images
- run ensemble prediction with multiple trained models

The main entry scripts are:

- `train.py`: model training
- `evaluate.py`: model evaluation
- `predict.py`: single-model inference
- `predict_ensemble.py`: ensemble inference
- `split_dataset.py`: sample-level dataset splitting

## 2. Repository Structure

```text
sentinel2_water_segmentation/
├── config/
├── dataset/
│   ├── Image/
│   ├── Mask/
│   └── README.md
├── models/
├── splits/
├── utils/
├── config.py
├── evaluate.py
├── predict.py
├── predict_ensemble.py
├── requirements.txt
├── train.py
└── README.md
```

## 3. Installation Instructions

### 3.1 Python Environment

A dedicated virtual environment is recommended.

Example with `venv`:

```bash
python -m venv .venv
```

Activate the environment:

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3.2 Install Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If GDAL or rasterio installation fails on your platform, install system-level geospatial libraries first, then reinstall the Python packages.

## 4. Documentation of Dependencies

The repository depends on the following software stack.

### 4.1 Core Language and Framework

- Python 3.8+
- PyTorch 2.0.1
- torchvision 0.15.2

### 4.2 Numerical and Machine Learning Packages

- numpy 1.24.3
- pandas 2.0.3
- scipy 1.11.1
- scikit-learn 1.3.0
- timm 0.9.7
- segmentation-models-pytorch 0.3.3
- transformers 4.31.0

### 4.3 Geospatial and Image I/O Packages

- GDAL 3.6.4
- rasterio 1.3.7
- geopandas 0.13.2
- pillow 10.0.0
- spectral 0.23.1
- opencv-python 4.8.0.74

### 4.4 Visualization and Utilities

- matplotlib 3.7.2
- seaborn 0.12.2
- tqdm 4.65.0
- pyyaml 6.0.1
- ttach 0.0.3

The exact pinned versions are listed in [requirements.txt](/f:/deeplearning/sentinel2_water_segmentation/requirements.txt).

## 5. Computational Requirements

### 5.1 Supported Hardware

The code can run on both GPU and CPU.

- GPU execution is recommended for training and large-scale inference
- CPU execution is possible for debugging, small-scale tests, and pipeline verification

### 5.2 Recommended Hardware

For typical use:

- GPU: NVIDIA GPU with CUDA support
- GPU memory: at least 8 GB recommended for training
- System memory: at least 16 GB RAM recommended
- Storage: enough disk space for GeoTIFF inputs, generated predictions, checkpoints, and copied split datasets

### 5.3 Practical Notes

- Training speed and feasible batch size depend on image tile size, number of channels, model architecture, and augmentation settings
- Large-image inference may require sliding-window prediction
- Ensemble inference requires additional memory because multiple models are loaded simultaneously

## 6. Data Availability Statement

This repository does not provide the full training dataset directly. The `dataset/` directory serves only as the default data structure for this project. Users must obtain the original data according to the instructions in [dataset/README.md](/f:/deeplearning/sentinel2_water_segmentation/dataset/README.md) and organize it into the corresponding directories.

### 6.1 Public Data Source

This project uses Sentinel-2 multispectral remote sensing images and corresponding binary water masks.

- Public dataset: `S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images`
  Download: https://github.com/MWieland/s1s2_water
  Description: S1S2-Water is a global reference dataset for training, validation, and testing convolutional neural networks for semantic segmentation of surface water bodies in publicly available Sentinel-1 and Sentinel-2 satellite images. The dataset consists of 65 triplets of Sentinel-1 and Sentinel-2 images with quality-checked binary water masks. Samples are drawn globally based on the Sentinel-2 tile grid (100 x 100 km), considering predominant land cover and availability of water bodies. Each sample is complemented with STAC-compliant metadata and a Digital Elevation Model (DEM) raster from the Copernicus DEM.

### 6.2 In-house Data Source

- In-house dataset: `Poyang Lake dataset`
  Status: not publicly available at present
  Contact: please contact the corresponding author for data-related inquiries

### 6.3 Clear Data-Sharing Statement

The full real research dataset used in the study cannot currently be shared through this repository. The repository provides the complete code pipeline, configuration files, and data organization rules so that the workflow itself can be reproduced. Users may validate the software pipeline with a minimal public or synthetic test case organized under `dataset/`.

## 7. Input Data Format

By default, the project expects the following structure:

```text
dataset/
├── Image/
├── Mask/
└── README.md
```

- `Image/`: input image tiles
- `Mask/`: corresponding binary water masks

Default image format:

- 6-band GeoTIFF
- band order: `B2, B3, B4, B8, B11, B12`

Mask format:

- single-band binary raster
- pixel values: `0` for non-water, `1` for water

## 8. File Naming Convention

`split_dataset.py` automatically matches images and masks by filename. The script supports:

- image names containing `_img_`, matched to `_msk_` or `_mask_`
- mask names of the form `xxx_msk.*` or `xxx_mask.*`
- file extensions `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`

Recommended naming convention:

```text
Image/710_img_1.tif
Mask/710_msk_1.tif
```

## 9. Dataset Splitting Principle

By default, the project uses `split_dataset.py` to perform random splitting at the sample level.

The implemented logic is:

1. scan all recognizable image files in `dataset/Image`
2. keep only samples with corresponding masks in `dataset/Mask`
3. shuffle valid samples using a fixed random seed
4. generate training, validation, and test lists according to the specified ratios

Default parameters:

- `val_ratio = 0.12`
- `test_ratio = 0.0`
- `seed = 42`

Default split proportions:

- training set: `88%`
- validation set: `12%`
- test set: `0%`

If an independent test set is required, explicitly set `--test_ratio`.

## 10. Basic Usage Instructions

### 10.1 Train a Single Model

Example:

```bash
python train.py --config config/aer_unet.yaml --model aer_unet
```

Other supported model names include:

- `unet`
- `deeplabv3_plus`
- `ultra_lightweight_deeplabv3_plus`

### 10.2 Evaluate a Trained Model

Example:

```bash
python evaluate.py --config config/aer_unet.yaml --model aer_unet --checkpoint_path path/to/checkpoint.pth --split val
```

### 10.3 Run Single-Model Inference

Example:

```bash
python predict.py --model aer_unet --checkpoint_path path/to/checkpoint.pth --input_dir dataset/Image --output_dir predictions
```

### 10.4 Run Ensemble Inference

Example:

```bash
python predict_ensemble.py --config config/improved_ensemble_config.yaml
```

Ensemble prediction requires correctly configured checkpoint paths and model settings in the corresponding YAML file.

## 11. Runnable Examples and Test Cases

The repository does not include the full study dataset. For software verification, users should prepare a minimal public, synthetic, or dummy test case in the default `dataset/` structure.

### 11.1 Generate Splits

```bash
python split_dataset.py --images_dir dataset/Image --masks_dir dataset/Mask --output_dir . --val_ratio 0.12 --test_ratio 0.0 --seed 42
```

This generates:

```text
splits/
├── train.txt
├── val.txt
└── test.txt
```

It also creates:

```text
datasets/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### 11.2 Minimal Evaluation Example

After preparing a minimal test case and a trained checkpoint:

```bash
python evaluate.py --config config/unet.yaml --model unet --checkpoint_path path/to/checkpoint.pth --split val
```

### 11.3 Minimal Prediction Example

```bash
python predict.py --model unet --checkpoint_path path/to/checkpoint.pth --input_dir dataset/Image --output_dir predictions
```

These commands are intended as pipeline verification examples. They confirm that the repository structure, model loading, raster I/O, and prediction outputs work correctly.

## 12. How-to Guide for Typical Use Cases

### 12.1 Typical Workflow for Water-Body Extraction

1. prepare image tiles and masks under `dataset/Image` and `dataset/Mask`
2. generate split files with `split_dataset.py`
3. select a model configuration from `config/`
4. train the model with `train.py`
5. evaluate the checkpoint with `evaluate.py`
6. run inference on new images with `predict.py`

### 12.2 How to Switch Models

To switch architectures:

- choose another YAML file in `config/`
- change the `--model` argument accordingly
- use the matching checkpoint trained for that architecture

Examples:

```bash
python train.py --config config/unet.yaml --model unet
python train.py --config config/deeplabv3_plus_baseline.yaml --model deeplabv3_plus
python train.py --config config/ultra_lightweight_deeplabv3_plus.yaml --model ultra_lightweight_deeplabv3_plus
```

### 12.3 How to Use Ensemble Strategies

The repository supports ensemble-based prediction through `predict_ensemble.py` and the corresponding utilities in `utils/`.

Typical steps:

1. train multiple models separately
2. prepare the ensemble YAML configuration
3. set checkpoint paths and strategy parameters
4. run `predict_ensemble.py`

Supported ensemble-related components include:

- basic ensemble loading
- weighted strategies
- adaptive weighting
- performance-weighted ensemble
- improved performance-weighted ensemble

## 13. User Guide: Inputs, Outputs, Options, and Expected Behaviour

### 13.1 Inputs

Expected input data:

- Sentinel-2 six-band image tiles
- optional binary masks for evaluation
- checkpoint files for trained models
- YAML configuration files in `config/`

### 13.2 Outputs

Depending on the script and options, the pipeline may produce:

- binary water masks
- probability maps
- evaluation metrics
- visualizations
- logs
- checkpoints

### 13.3 Common Configurable Options

Frequently used options include:

- thresholding
  - `--threshold`
  - controls conversion from probability map to binary water mask
- sliding-window inference
  - `--use_sliding_window`
  - `--tile_size`
  - `--overlap`
- post-processing
  - median filtering
  - Gaussian smoothing
  - morphological closing/opening
  - minimum object size filtering
  - hole removal
  - optional CRF refinement
- split selection
  - `--split val`
  - `--split test`

### 13.4 Expected Behaviour

When the pipeline is configured correctly:

- images are loaded as 6-band raster inputs
- model outputs are converted to water probability maps
- thresholds and optional post-processing produce a binary water mask
- evaluation computes standard segmentation metrics such as IoU and Dice
- ensemble prediction combines outputs from multiple checkpoints according to the selected strategy

## 14. Reproducibility of Main Results

### 14.1 What Can Be Reproduced

This repository provides the code needed to reproduce the full processing pipeline:

- dataset organization
- sample matching
- data splitting
- model training
- model evaluation
- single-model inference
- ensemble inference

### 14.2 What Cannot Be Fully Reproduced from the Repository Alone

The exact main results reported in the study depend on the full real dataset, including the non-public in-house Poyang Lake dataset. Because the full research dataset is not publicly released, exact numeric reproduction of all paper results is not possible from this repository alone.

### 14.3 What Users Can Reproduce

Users can still reproduce the computational workflow by:

1. obtaining the public S1S2-Water dataset
2. organizing public or synthetic sample data under `dataset/`
3. generating splits with `split_dataset.py`
4. running training, evaluation, and prediction scripts with the provided configurations

This allows reproducible verification of the implemented pipeline even when the full real research dataset cannot be shared.

## 15. Correspondence with Configuration Files

The YAML configuration files under `config/` use the following default paths:

- image directory: `dataset/Image`
- mask directory: `dataset/Mask`
- split files: `splits/train.txt`, `splits/val.txt`, `splits/test.txt`

If you change the data storage locations, update the corresponding YAML files accordingly.

## 16. Notes for Publication

For journal or archive submission, the following statements are recommended and already reflected in this README:

- the code pipeline is publicly available
- the full real research dataset is not publicly shared
- public data sources are cited
- a minimal public or synthetic verification case may be used to validate the code path
- exact numerical reproduction of the paper's final results requires access to the full study dataset
