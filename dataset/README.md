This repository does not provide the full training dataset directly. The dataset/ directory serves only as the default data structure for this project. Users must obtain the original data according to the instructions below and organize it into the corresponding directories.
1. Data Sources
This project uses Sentinel-2 multispectral remote sensing images and their corresponding binary water body annotation masks.
Public Dataset: S1S2-Water: A global dataset for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2 satellite images
Download: https://github.com/MWieland/s1s2_water
Description: S1S2-Water dataset is a global reference dataset for training, validation and testing of convolutional neural networks for semantic segmentation of surface water bodies in publicly available Sentinel-1 and Sentinel-2 satellite images. The dataset consists of 65 triplets of Sentinel-1 and Sentinel-2 images with quality checked binary water mask. Samples are drawn globally on the basis of the Sentinel-2 tile-grid (100 x 100 km) under consideration of predominant landcover and availability of water bodies. Each sample is complemented with STAC-compliant metadata and Digital Elevation Model (DEM) raster from the Copernicus DEM.
In-house Dataset: Poyang Lake dataset
Note: Not publicly available temporarily. For inquiries, please contact the corresponding author.
2. Directory Structure
The project reads data in the following structure by default:
text
dataset/
├── Image/
├── Mask/
└── README.md
Image/: Stores input images, default 6-band GeoTIFF with band order B2, B3, B4, B8, B11, B12
Mask/: Stores corresponding binary water body masks with pixel values 0/1
3. File Naming Convention
split_dataset.py automatically matches images and masks by filename. The script supports the following naming patterns:
Images containing _img_ will be matched with corresponding _msk_ or _mask_
Also supports xxx_msk.*, xxx_mask.*
Supported extensions: .tif, .tiff, .png, .jpg, .jpeg
The following consistent format is recommended for reliability:
text
Image/710_img_1.tif
Mask/710_msk_1.tif
4. Dataset Splitting Principle
By default, the project uses split_dataset.py to perform random splitting at the sample level, following the same logic as the code:
Scan all recognizable image files in Image/
Keep only samples with corresponding masks in Mask/
Shuffle with a fixed random seed
Generate training, validation, and test set lists by ratio
Default parameters of the script:
val_ratio = 0.12
test_ratio = 0.0
seed = 42
By default, all valid samples are randomly split into:
Training set: 88%
Validation set: 12%
Test set: 0%
To use an independent test set, set --test_ratio explicitly.
5. Splitting Command
Run in the project root directory:
bash
运行
python split_dataset.py --images_dir dataset/Image --masks_dir dataset/Mask --output_dir . --val_ratio 0.12 --test_ratio 0.0 --seed 42
The following files will be generated:
text
splits/
├── train.txt
├── val.txt
└── test.txt
The script also copies files into an additional datasets/ structure organized by train/val/test for direct use in some training pipelines:
text
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
6. Correspondence with Configuration Files
The project’s YAML configuration files use the following default paths:
Image directory: dataset/Image
Mask directory: dataset/Mask
Split files: splits/train.txt, splits/val.txt, splits/test.txt
If you change the data storage location, update the corresponding YAML configurations under config/ accordingly.
