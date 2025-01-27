# Framework for the evaluation of XAI algorithms (XAIEV)

## Installation (work in progress)

- clone the repo
- `pip install -e .`
- ask the authors for the dataset


## Usage

### General Notes on Paths

Many scripts and notebooks in this repo depend on paths. To ensure that the code runs on different machines (local development machines, HPC, etc) we use a `.env` file. This file is machine-specific and is expected to define the necessary paths in environment variables.

Example (see also .env-example):

```.env
# Note: This directory might contain several GB of (auto-generated) data
XAIEV_BASE_DIR="/home/username/xaiev/data"
```

This file is evaluated by `utils.read_paths_from_dotenv()`. Note: The package `opencv-python` has to be installed (see `requirements.txt`)


The expected path structure is as follows:

```
<BASE_DIR>                      specified in .env file
├── atsds_large/
│   ├── test/
│   │   ├── 0001/               class directory
│   │   │   ├── 000000.png      individual image of this class
│   │   │   └── ...             more images
│   │   └── ...                 more classes
│   └── train/
│       └── <class dirs with image files>
│
├── atsds_large_background/...  background images with same structure
│                               as in atsds_large (test/..., train/...)
│
├── atsds_large_mask/...        corresponding mask images with same structure
│                               as in atsds_large (test/..., train/...)
├── model_checkpoints/
│   ├── convnext_tiny_1_1.tar
│   ├── resnet50_1_1.tar
│   ├── simple_cnn_1_1.tar
│   └── vgg16_1_1.tar
│
├── XAI_evaluation
│   ├── simple_cnn/gradcam/test/    same structure as `XAI_results`
│   │   ├── revelation
│   │   └── occlusion
│   └── ...                     other XAI methods and models
│
└── XAI_results
    ├── simple_cnn/             cnn model directory
    │   ├── gradcam/            xai method
    │   │   ├── test/           split fraction (train/test)
    │   │   │   ├── mask/
    │   │   │   │   ├── 000000.png.npy
    │   │   │   │   └── ...
    │   │   │   ├── mask_on_image/
    │   │   │   │   ├── 000000.png
    │   │   │   │   └── ...
    │   …   …   …
    ├── vgg16/...
    ├── resnet50/..
    ├── convnext_tiny/..
```
