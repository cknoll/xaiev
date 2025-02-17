### Standard libraries
import argparse
import os
import random

### 3rd party libraries
import numpy as np
import cv2
from PIL import Image
# from captum.attr import IntegratedGradients
# from captum.attr import visualization as viz

## PyTorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F

## Local libraries
from .utilmethods import get_default_arg_parser, read_conf_from_dotenv, setup_environment, prepare_categories_and_images, create_output_directories, save_xai_outputs, normalize_image, get_rgb_heatmap
from .ATSDS import ATSDS
from .model import get_model, load_model
from .integrated_gradients import get_ig_attributions
from . import utils

pjoin = os.path.join

TRANSFORM_TEST = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def compute_ig_masks(model: torch.nn.Module, device: torch.device, categories: list[str], 
                       imagedict: dict[str, list[str]], label_idx_dict: dict[str, int], 
                       output_path: str, images_path: str, runs: int = 64) -> None:
    """
    Generate Integrated Gradients (IG) visualizations for each image in the dataset and save them.

    Args:
        model (torch.nn.Module): The model used for generating IG visualizations.
        device (torch.device): The device to run the model on (GPU or CPU).
        categories (list): List of categories in the dataset.
        imagedict (dict): A dictionary of image filenames for each category.
        label_idx_dict (dict): A dictionary mapping category names to indices.
        output_path (str): Path where IG results will be saved.
        images_path (str): Path to the dataset images.
        runs (int, optional): Number of IG runs for smoothing. Defaults to 64.
    """
    for category in categories:
        images = imagedict[category]
        for image_name in images:
            with Image.open(os.path.join(images_path, category, image_name)) as img:
                # Preprocess the image to get the input tensor
                current_image_tensor = TRANSFORM_TEST(img).unsqueeze(0).to(device)

                # Perform Integrated Gradients computation using int_g.get_ig_attributions
                ig_attributions = get_ig_attributions(
                    model=model,
                    image_tensor=current_image_tensor,
                    label_idx=label_idx_dict[category],
                    runs=runs
                )
                # ig_attributions = ig_attributions.squeeze().detach().cpu().numpy()

                # Convert to visualization format
                ig_mask = np.sum(ig_attributions, axis=0)  # Aggregate across channels
                ig_mask = normalize_image(ig_mask)

                # Overlay IG mask on original image
                overlay_image = np.array(img).astype(np.float32) / 255.0
                mask_on_image_result = mask_on_image_ig(ig_mask, overlay_image, alpha=0.3)

                # Create output directories if they do not exist
                mask_output_dir = os.path.join(output_path, category, 'mask')
                overlay_output_dir = os.path.join(output_path, category, 'mask_on_image')
                os.makedirs(mask_output_dir, exist_ok=True)
                os.makedirs(overlay_output_dir, exist_ok=True)

                # Save IG mask and overlay image
                mask_output_path = os.path.join(mask_output_dir, image_name.replace('.PNG', '.npy'))
                overlay_output_path = os.path.join(overlay_output_dir, image_name)
                np.save(mask_output_path, ig_mask)
                Image.fromarray((mask_on_image_result * 255).astype(np.uint8)).save(overlay_output_path, "PNG")


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
    output_path = pjoin(BASE_DIR, "XAI_results", model_name, "gradcam", dataset_split)

    # Setup environment
    device = setup_environment(random_seed)

    # Load dataset and dataloader
    testset = ATSDS(root= BASE_DIR, split=dataset_split, dataset_type=dataset_type, transform=TRANSFORM_TEST)

    # Load model
    model = get_model(model_name, n_classes = testset.get_num_classes())
    model = model.to(device)
    model.eval()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Load checkpoint
    epoch,trainstats = load_model(model, optimizer, scheduler, os.path.join(CHECKPOINT_PATH, model_cpt), device)
    print(f"Model checkpoint loaded. Epoch: {epoch}")

    # Prepare categories and images
    categories, label_idx_dict, imagedict = prepare_categories_and_images(IMAGES_PATH)

    # Ensure output directories exist
    create_output_directories(output_path, categories)

    # Generate Integrated Gradients visualizations
    compute_ig_masks(model, device, categories, imagedict, label_idx_dict, output_path, IMAGES_PATH)


# if __name__ == "__main__":
#     main()
