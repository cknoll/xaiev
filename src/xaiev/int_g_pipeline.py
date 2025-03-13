### Standard libraries
import argparse
import os
import random
import pickle

### 3rd party libraries
import numpy as np
import cv2
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

## PyTorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

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
from .integrated_gradients import get_ig_attributions
from . import utils

pjoin = os.path.join

TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def compute_ig_masks(model, device, categories, imagedict, label_idx_dict, output_path, images_path):
    """Generate Integrated Gradients visualizations for each image in the dataset."""
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    for category in categories:
        images = imagedict[category]
        for image_name in images:
            image_path = os.path.join(images_path, category, image_name)
            with open(image_path, "rb") as f:
                with Image.open(f) as current_image:
                    current_image_tensor = TRANSFORM_TEST(current_image).unsqueeze(0).to(device)
                    current_image_tensor.requires_grad = True

                    # Set baseline as a tensor of zeros
                    baseline = torch.zeros_like(current_image_tensor)

                    # Perform Integrated Gradients computation
                    attr_ig = ig.attribute(
                        current_image_tensor, baselines=baseline, target=label_idx_dict[category]
                    )
                    attr_ig = attr_ig.squeeze().detach().cpu().numpy()

                    # Convert to visualization format
                    ig_mask = np.sum(attr_ig, axis=0)  # Aggregate across channels
                    ig_mask = normalize_image(ig_mask)

                    # Overlay IG mask on original image
                    overlay_image = np.array(current_image).astype(np.float32) / 255.0
                    mask_on_image_result = mask_on_image_ig(ig_mask, overlay_image, alpha=0.3)

                    # Create output directories if they do not exist
                    mask_output_dir = os.path.join(output_path, category, "mask")
                    overlay_output_dir = os.path.join(output_path, category, "mask_on_image")
                    os.makedirs(mask_output_dir, exist_ok=True)
                    os.makedirs(overlay_output_dir, exist_ok=True)

                    # Save IG mask and overlay image
                    mask_output_path = os.path.join(mask_output_dir, image_name.replace(".jpg", ".npy"))
                    overlay_output_path = os.path.join(overlay_output_dir, image_name)
                    np.save(mask_output_path, ig_mask)
                    Image.fromarray((mask_on_image_result * 255).astype(np.uint8)).save(
                        overlay_output_path, "PNG"
                    )


def mask_on_image_ig(mask, img, alpha=0.5):
    # Ensure the mask and image have the same dimensions
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Generate heatmap from the mask
    heatmap = get_rgb_heatmap(mask)

    # Squeeze image if it has extra dimensions
    if len(img.shape) == 4 and img.shape[0] == 1:  # Batch size of 1
        img = img.squeeze()

    # Normalize the image to [0, 1] if it's not already
    if img.max() > 1:
        img = img.astype(np.float32) / 255

    # Blend the heatmap and image
    cam_on_img = (1 - alpha) * img + alpha * heatmap
    return np.copy(cam_on_img)


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
    output_path = pjoin(BASE_DIR, "XAI_results", model_name, "ig", dataset_split)

    # Setup environment
    device = setup_environment(random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root=BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(model_name, n_classes=testset.get_num_classes())
    model = model.to(device)
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5000)

    # Load checkpoint
    epoch, trainstats = load_model(model, optimizer, scheduler, pjoin(CHECKPOINT_PATH, model_cpt), device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    # Ensure output directories exist
    create_output_directories(output_path, categories)

    # Generate Integrated Gradients visualizations
    compute_ig_masks(model, device, categories, imagedict, label_idx_dict, output_path, IMAGES_PATH)
