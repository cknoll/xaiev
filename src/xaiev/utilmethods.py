import argparse
import os
import random
from PIL import Image
from types import SimpleNamespace  # used as flexible Container Class

import torch

import numpy as np
import cv2
from dotenv import load_dotenv

from . import utils


def mask_on_image(mask: np.ndarray, img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay an XAI mask on the original image.

    Args:
        mask (numpy.ndarray): The XAI-CAM mask.
        img (numpy.ndarray): The original image.
        alpha (float, optional): The transparency of the mask overlay. Default is 0.5.

    Returns:
        numpy.ndarray: The image with the XAI-CAM mask overlaid.
    """
    heatmap = get_rgb_heatmap(mask)
    img = img.squeeze()
    cam_on_img = (1 - alpha) * img + alpha * heatmap
    return np.copy(cam_on_img)


def get_rgb_heatmap(mask: np.ndarray) -> np.ndarray:
    """Convert mask to RGB heatmap."""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    return np.copy(heatmap)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image values to [0, 1]."""
    return np.nan_to_num((img - img.min()) / (img.max() - img.min()), nan=0.0, posinf=0.0, neginf=0.0)


def setup_environment(seed: int) -> torch.device:
    """
    Set up the environment for reproducibility and specify the device (GPU or CPU).

    Args:
        seed (int): The seed value for random number generation, ensuring reproducibility.

    Returns:
        device (torch.device): The device (either GPU or CPU) to run the model on.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


# get a cutout based on a cutoff value
def get_cutoff_area(mask, img, cutoff=0.5):
    for i in range(3):
        img[:, :, i] = np.where(mask > cutoff, img[:, :, i], 0)
    return np.copy(img)


# get a cutout based on a percentage value.
def get_percentage_of_image(image, mask, percentage, fill_value=0.0):
    masked_image = np.zeros_like(image)
    n = mask.size
    sorted_values = np.sort(mask.flatten("K"))[::-1]

    index = int(n / 100 * percentage)
    index_2 = n // 100 * percentage
    cutoff = sorted_values[index]
    for i in range(3):
        masked_image[:, :, i] = np.where(mask - cutoff > 0.0, image[:, :, i], fill_value)
    return masked_image


def get_percentage_of_image_1d(image, mask, percentage, fill_value=0.0):
    image = normalize_image(image)
    mask = normalize_image(mask)
    masked_image = np.zeros_like(image)
    n = mask.size
    sorted_values = np.sort(mask.flatten("K"))[::-1]

    index = int(n / 100 * percentage)
    index_2 = n // 100 * percentage
    cutoff = sorted_values[index]
    for i in range(3):
        masked_image = np.where(mask - cutoff > 0.0, image, fill_value)
    return masked_image


def get_contained_part(mask1, mask2):
    mask1, mask2 = normalize_image(mask1), normalize_image(mask2)
    return np.array((mask1 == 1.0) & (mask2 == 1.0))


def get_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model_full_name", "-n", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)"
    )
    parser.add_argument("--model_cp_base_path", "-cp", type=str, help="directory of model checkpoints")
    parser.add_argument("--data_base_path", "-d", type=str, help="data path")
    # parser.add_argument('--output_path', type=str, default="data/XAI_results/", help="Path to save outputs.")
    parser.add_argument("--dataset_type", type=str, default="atsds_large", help="Type of the dataset.")
    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split (e.g., 'train', 'test')."
    )
    parser.add_argument("--random_seed", type=int, default=1414, help="Random seed for reproducibility.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for data loader.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading.")
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of images for PRISM explanation."
    )

    return parser


# TODO: rename "category" to "class_" etc.
def generate_adversarial_examples(
    adv_folder,
    pct_range,
    categories,
    image_dict,
    img_path,
    background_dir,
    xai_dir,
    mask_condition,
    limit: int,
):
    """
    Generate adversarial examples based on the given mask condition.

    """

    if limit is not None:
        assert isinstance(limit, int) and limit > 0
    else:
        # Note: `None` is a valid upper bound (-> no restriction)
        pass

    for pct in pct_range:
        print(f"Processing percentage: {pct / 10}%")
        # note: we do not apply the limit here because the trained model requires all classes to be present
        for category in categories:
            output_dir = os.path.join(adv_folder, str(pct), category)
            os.makedirs(output_dir, exist_ok=True)

            images = image_dict[category]

            for imagename in images[:limit]:
                # Load original image and background
                current_img = normalize_image(
                    np.array(Image.open(os.path.join(img_path, category, imagename)))
                )
                current_background = normalize_image(
                    np.array(Image.open(os.path.join(background_dir, category, imagename)))
                )

                # Load and process XAI mask
                xai_mask = np.load(os.path.join(xai_dir, category, "mask", f"{imagename}.npy"))
                adv_mask = normalize_image(
                    get_percentage_of_image(np.ones_like(current_img), xai_mask, pct / 10)
                )

                # Create adversarial example using the mask condition
                adv_example = np.where(mask_condition(adv_mask), current_img, current_background)
                adv_example_save = Image.fromarray((adv_example * 255).astype("uint8"))

                # Save adversarial example
                adv_example_save.save(os.path.join(output_dir, imagename))


def create_image_dict(BASE_DIR, DATASET, DATASET_SPLIT):
    IMAGES_PATH = os.path.join(BASE_DIR, DATASET, DATASET_SPLIT)

    # Define our Categories
    CATEGORIES = sorted(os.listdir(IMAGES_PATH))

    image_dict = {}
    for cat in CATEGORIES:
        image_dict[cat] = []
        imagelist = os.listdir(os.path.join(IMAGES_PATH, cat))
        for im in imagelist:
            image_dict[cat].append(im)

    return CATEGORIES, image_dict


def read_conf_from_dotenv() -> SimpleNamespace:
    assert os.path.isfile(".env")
    load_dotenv()

    conf = SimpleNamespace()
    conf.BASE_DIR = os.getenv("BASE_DIR")
    conf.MODEL_DIR = os.getenv("MODEL_DIR")

    assert conf.BASE_DIR is not None
    assert conf.MODEL_DIR is not None

    return conf


def get_dir_path(*parts, check_exists=True):
    path = os.path.join(*parts)

    if check_exists and not os.path.isdir(path):
        msg = f"Path {path} unexpectedly is not a directory"
        raise FileNotFoundError(msg)
    return path


def prepare_categories_and_images(image_path: str) -> tuple[list[str], dict[str, int], dict[str, list[str]]]:
    """
    Prepare categories and image file lists for each category in the dataset.

    Args:
        image_path (str): Path to the root directory of images.

    Returns:
        categories (list): List of category names (subdirectories in the image path).
        label_idx_dict (dict): A dictionary mapping category names to indices.
        image_dict (dict): A dictionary mapping categories to lists of image filenames.
    """
    categories = sorted(os.listdir(image_path))
    label_idx_dict = {cat: idx for idx, cat in enumerate(categories)}
    image_dict = {cat: os.listdir(os.path.join(image_path, cat)) for cat in categories}
    return categories, label_idx_dict, image_dict


def create_output_directories(output_path: str, categories: list[str]) -> None:
    """
    Create the necessary output directories to store the results.

    Args:
        output_path (str): The root directory where results will be saved.
        categories (list): List of category names for which to create subdirectories.
    """
    os.makedirs(output_path, exist_ok=True)
    for category in categories:
        for output_type in ["mask", "mask_on_image"]:
            os.makedirs(os.path.join(output_path, category, output_type), exist_ok=True)


def save_xai_outputs(
    mask: np.ndarray, original_image: np.ndarray, category: str, image_name: str, output_path: str
) -> None:
    """
    General function to save the mask and overlay image generated by any XAI method.

    Args:
        mask (numpy.ndarray): The mask to save.
        original_image (numpy.ndarray): The original image to overlay the mask on.
        category (str): The category of the image.
        image_name (str): The filename of the image.
        output_path (str): Path to save the outputs.
    """
    mask = normalize_image(mask)
    overlay_image = mask_on_image(mask, normalize_image(original_image), alpha=0.3)
    overlay_image = (overlay_image * 255).astype(np.uint8)

    mask_path = os.path.join(output_path, category, "mask", image_name)
    overlay_path = os.path.join(output_path, category, "mask_on_image", image_name)

    np.save(mask_path, mask)
    Image.fromarray(overlay_image).save(overlay_path, "PNG")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: torch.device,
) -> tuple[int, dict]:
    """
    Load a model checkpoint from the specified file and restore model, optimizer, and scheduler states.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state_dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state_dict into.
        scheduler (torch.optim.lr_scheduler): The scheduler to load the state_dict into.
        device (torch.device): The device to load the checkpoint on.

    Returns:
        epoch (int): The epoch the model was at when the checkpoint was saved.
        trainstats (dict): The training statistics stored in the checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"], checkpoint["trainstats"]
