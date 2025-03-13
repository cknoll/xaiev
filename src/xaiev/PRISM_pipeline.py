## Standard libraries
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt

##PyTorch
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

## 3rd party libraries
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchprism import PRISM

## Local libraries
from .utilmethods import (
    get_default_arg_parser,
    read_conf_from_dotenv,
    setup_environment,
    prepare_categories_and_images,
    create_output_directories,
    save_xai_outputs,
    normalize_image,
    get_rgb_heatmap,
)
from .ATSDS import ATSDS
from .model import get_model, load_model
from . import utils

pjoin = os.path.join

TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def generate_prism_visualizations(
    model: torch.nn.Module,
    device: torch.device,
    categories: list[str],
    imagedict: dict[str, list[str]],
    label_idx_dict: dict[str, int],
    output_path: str,
    images_path: str,
) -> None:
    """
    Generate PRISM visualizations for each image in the dataset and save them.

    Args:
        model (torch.nn.Module): The model used for generating PRISM visualizations.
        device (torch.device): The device to run the model on (GPU or CPU).
        categories (list): List of categories in the dataset.
        imagedict (dict): A dictionary of image filenames for each category.
        label_idx_dict (dict): A dictionary mapping category names to indices.
        output_path (str): Path where PRISM results will be saved.
        images_path (str): Path to the dataset images.
    """
    # Register hooks for PRISM
    PRISM.prune_old_hooks(model)
    PRISM.register_hooks(model)
    print("Hooks registered:", any(module._forward_hooks for module in model.modules()))
    # print(PRISM.get_maps())
    for category in categories:
        images = imagedict[category]
        for image_name in images:
            with Image.open(pjoin(images_path, category, image_name)) as img:
                current_image_tensor = TRANSFORM_TEST(img).unsqueeze(0).to(device)

                # Forward pass to generate PRISM maps
                model(current_image_tensor)

                # Retrieve PRISM maps
                prism_maps = PRISM.get_maps()
                drawable_prism_maps = prism_maps.permute(0, 2, 3, 1).detach().cpu().numpy().squeeze()

                # Save PRISM maps overlayed on original image
                save_moi = Image.fromarray((drawable_prism_maps * 255).astype(np.uint8))
                save_moi.save(pjoin(output_path, category, "mask_on_image", image_name), "PNG")

                # Normalize and save PRISM mask
                mask_raw = np.array(save_moi)
                mask_norm = (mask_raw - mask_raw.mean()) // mask_raw.std()
                mask_f = np.max(mask_norm, axis=2)
                mask = normalize_image(
                    F.interpolate(
                        F.relu(torch.Tensor(mask_f)).reshape(1, 1, 224, 224), (512, 512), mode="bilinear"
                    )
                    .squeeze()
                    .squeeze()
                    .numpy()
                )
                # smooth heatmap
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pooled_mask = avg_pooling(mask_tensor, kernel_size=129, stride=1)

                grad_mask = normalize_image(mask) + normalize_image(pooled_mask.squeeze().numpy()) / 100

                np.save(pjoin(output_path, category, "mask", image_name), grad_mask)


def avg_pooling(mask: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Apply average pooling to a tensor.

    Args:
        mask (torch.Tensor): The input tensor to pool.
        kernel_size (int): Size of the pooling kernel. Default is 129.
        stride (int): Stride of the pooling operation. Default is 1.

    Returns:
        torch.Tensor: The pooled tensor.
    """
    pooling = torch.nn.AvgPool2d(
        kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, count_include_pad=False
    )
    return pooling(mask)


def main(model_full_name, conf: utils.CONF):

    BASE_DIR = conf.XAIEV_BASE_DIR
    CHECKPOINT_PATH = conf.MODEL_CP_PATH

    # Changable Parameters
    model_name = "_".join(model_full_name.split("_")[:-2])
    model_cpt = model_full_name + ".tar"

    dataset_type = conf.DATASET_NAME
    dataset_split = conf.DATASET_SPLIT
    random_seed = conf.RANDOM_SEED

    IMAGES_PATH = pjoin(BASE_DIR, dataset_type, dataset_split)
    output_path = pjoin(BASE_DIR, "XAI_results", model_name, "prism", dataset_split)

    # Setup environment
    device = setup_environment(random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root=BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(model_name, n_classes=testset.get_num_classes())
    model = model.to(device)
    model.eval()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Load checkpoint
    epoch, trainstats = load_model(model, optimizer, scheduler, pjoin(CHECKPOINT_PATH, model_cpt), device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    # Ensure output directories exist
    create_output_directories(output_path, categories)

    # Generate PRISM visualizations
    generate_prism_visualizations(
        model, device, categories, imagedict, label_idx_dict, output_path, IMAGES_PATH
    )
