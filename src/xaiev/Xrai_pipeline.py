## Standard libraries
import argparse
import os

## 3rd party libraries
import pickle
import numpy as np
from PIL import Image
import saliency.core as saliency
import cv2

##PyTorch
import torch
import torch.nn.functional as F
from torchvision import transforms

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


def generate_xrai_visualizations(
    model: torch.nn.Module,
    device: torch.device,
    categories: list[str],
    imagedict: dict[str, list[str]],
    label_idx_dict: dict[str, int],
    output_path: str,
    images_path: str,
) -> None:
    """
    Generate XRAI visualizations for each image in the dataset using precomputed IG masks.

    Args:
        model (torch.nn.Module): The model used for generating XRAI visualizations.
        device (torch.device): The device to run the model on (GPU or CPU).
        categories (list): List of categories in the dataset.
        imagedict (dict): A dictionary of image filenames for each category.
        label_idx_dict (dict): A dictionary mapping category names to indices.
        output_path (str): Path where XRAI results will be saved.
        images_path (str): Path to the dataset images.
    """

    # Initialize the XRAI object
    xrai_obj = saliency.XRAI()

    for category in categories:
        images = imagedict[category]
        for image_name in images:
            ig_path = pjoin(output_path.replace("xrai", "ig"), category, "mask", image_name + ".npy")

            # Check if IG mask exists
            if not os.path.exists(ig_path):
                print(f"IG mask not found for {image_name}. Skipping...")
                continue

            # Open and process the image
            with Image.open(pjoin(images_path, category, image_name)) as img:
                current_image_tensor = TRANSFORM_TEST(img).unsqueeze(0).to(device)

                # Convert current_image_tensor to (H, W, C)
                current_image_np = np.moveaxis(current_image_tensor.squeeze(0).cpu().numpy(), 0, -1)

                # Load the IG mask
                ig_attribs = np.load(ig_path)
                mask_raw = xrai_obj.GetMask(
                    current_image_np,
                    None,
                    base_attribution=np.repeat(ig_attribs[:, :, np.newaxis], 3, axis=2),
                )

                # Normalize the mask and resize
                mask = normalize_image(
                    F.interpolate(
                        torch.Tensor(mask_raw).unsqueeze(0).unsqueeze(0), size=(512, 512), mode="bilinear"
                    )
                    .squeeze()
                    .numpy()
                )

                # smooth heatmap
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pooled_mask = avg_pooling(mask_tensor, kernel_size=129, stride=1)

                grad_mask = normalize_image(mask) + normalize_image(pooled_mask.squeeze().numpy()) / 100
                np.save(pjoin(output_path, category, "mask", image_name), grad_mask)

                # Overlay XRAI mask on the original image
                overlay_image = mask_on_image_ig(normalize_image(mask), normalize_image(np.array(img)))
                overlay_output_path = pjoin(output_path, category, "mask_on_image", image_name)
                Image.fromarray((overlay_image * 255).astype(np.uint8)).save(overlay_output_path, "PNG")


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
    output_path = pjoin(BASE_DIR, "XAI_results", model_name, "xrai", dataset_split)

    # Setup environment
    device = setup_environment(random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root=BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(model_name, n_classes=testset.get_num_classes())
    model = model.to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Load checkpoint
    epoch, trainstats = load_model(model, optimizer, scheduler, pjoin(CHECKPOINT_PATH, model_cpt), device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    # Ensure output directories exist
    create_output_directories(output_path, categories)

    # Generate XRAI visualizations
    generate_xrai_visualizations(
        model, device, categories, imagedict, label_idx_dict, output_path, IMAGES_PATH
    )
