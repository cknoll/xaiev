[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/cknoll/xaiev/actions/workflows/python-app.yml/badge.svg)](https://github.com/cknoll/xaiev/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/xaiev.svg)](https://pypi.org/project/xaiev/)

# Framework for the Evaluation of XAI Algorithms (XAIEV)

**This code is heavily based on the [master thesis](https://github.com/Lunnaris01/Masterarbeit_Public) of Julian Ulrich [@Lunnaris01](https://github.com/Lunnaris01/).**

## Installation (work in progress)

- clone the repo
- `pip install -e .`
- download the "atsds_large" dataset from [here](https://datashare.tu-dresden.de/s/4mNxbpx343Pq835)


## Usage

### Bootstrap

- Open terminal in the directory you want to use for future xaiev-usage.
- Run `xaiev --bootstrap`.
    - This creates `.env` file in current working directory.
- Edit this file (see next section).

### General Notes on Paths

Many scripts and notebooks in this repo depend on paths. To ensure that the code runs on different machines (local development machines, HPC, etc) we use a `.env` file. This file is machine-specific and is expected to define the necessary paths in environment variables.

Example (see also .env-example):

```.env
# Note: This directory might contain several GB of (auto-generated) data
XAIEV_BASE_DIR="/home/username/xaiev/data/atsds_large"
```

This file is evaluated by `utils.read_paths_from_dotenv()`.


The expected path structure is as follows (as shown in the dataset "atsds_large"):

```
<XAIEV_BASE_DIR>                      xaiev directory for one dataset (e.g.
│                               atsds_large) specified in .env file
├── imgs_main/                  main images (not masks, not background)
│   ├── test/
│   │   ├── 0001/               class directory
│   │   │   ├── 000000.png      individual image of this class
│   │   │   └── ...             more images
│   │   └── ...                 more classes
│   └── train/
│       └── <class dirs with image files>
│
├── imgs_background/...         background images; same structure
│                               as in imgs_main (test/..., train/...)
│
├── imgs_mask/...               corresponding mask images with same structure
│
├── model_checkpoints/          saved checkpoints for models trained on this
│   │                           dataset
│   ├── convnext_tiny_1_1.tar
│   ├── resnet50_1_1.tar
│   ├── simple_cnn_1_1.tar
│   └── vgg16_1_1.tar
│                               as in imgs_main (test/..., train/...)
├── inference/
│   ├── images_to_classify      directory for images which should be classified
│   └── classification_results
│       ├── simple_cnn_1_1      classification results for a specific model
│       └── ...
│
├── XAI_results
│   ├── simple_cnn/             directory specifying cnn model
│   │   ├── gradcam/            xai method
│   │   │   ├── test/           split fraction (train/test)
│   │   │   │   ├── mask/
│   │   │   │   │   ├── 000000.png.npy
│   │   │   │   │   └── ...
│   │   │   │   ├── mask_on_image/
│   │   │   │   │   ├── 000000.png
│   │   │   │   │   └── ...
│   │   …   …   …
│   ├── vgg16/...
│   ├── resnet50/...
│   └── convnext_tiny/...
│
├── XAI_evaluation
│   ├── simple_cnn/gradcam/test/    same structure as `XAI_results`
│   │           ├── revelation
│   │           │   └── results.pcl
│   │           └── occlusion
│   │               └── results.pcl
│   └── ...                     other XAI methods and models
└── ...
```


### General Usage

The four steps of the pipeline (with example calls):
- (1) model training,
    - `xaiev train --model simple_cnn_1_1`
- (2) applying XAI algorithms to generate weighted saliency maps,
    - `xaiev create-saliency-maps --xai-method gradcam --model simple_cnn_1_1`
    - `xaiev create-saliency-maps --xai-method int_g --model simple_cnn_1_1`
- (3) generating new test images with varying percentages of "important" pixels removed or retained, and
    - `xaiev create-eval-images --xai-method gradcam --model simple_cnn_1_1`
- (4) statistically evaluating accuracy changes on these test images and comparison to the ground truth.
    - `xaiev eval --xai-method gradcam --model simple_cnn_1_1`

#### Arguments for `xaiev create-saliency-maps`
- (1) **`--xai-method`** (required):
  Selects the explainable AI (XAI) method to be used in the analysis.
  **Example:**
  `--xai-method gradcam`

- (2) **`--model`** (required):
  Specifies the full model name.
  **Example:**
  `--model simple_cnn_1_1`

- (3) **`--dataset_split`** (optional):
  Indicates which dataset part to use (e.g., `train` or `test`; see expected directory layout above).
  **Default:** `test`
  **Example:**
  `--dataset_split test`

- (4) **`--random_seed`** (optional):
  An integer used to set the random seed for reproducibility.
  **Default:** `1414`
  **Example:**
  `--random_seed 1414`

#### Additional calls:

- Use a trained model to perform classification
    - `xaiev inference --model simple_cnn_1_1`

## Contributing and Development Notes

This software is provided in the hope that it will be useful. If you spot opportunities for improvement feel free to open an issue or even better a pull-request.


### Unittests and Continuous Integration (CI)

- Unittests can be run by `pytest`.
- The CI runs the important steps of the pipeline based on the drastically reduced dataset [atsds-demo](https://github.com/cknoll/atsds-demo). See [.github/workflows/python-app.yml](.github/workflows/python-app.yml) for details.


### Code Style

- We (aim to) use `black -l 110 ./` to ensure coding style consistency, see also: [code style black](https://github.com/psf/black).
- We recommend using [typing hints](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
