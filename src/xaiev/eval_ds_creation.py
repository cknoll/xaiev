import os
from ipydex import IPS

from . import utils
from .utilmethods import get_dir_path, create_image_dict, generate_adversarial_examples


def create_occlusion_dataset(conf: utils.CONF):
    occlusion_condition = lambda adv_mask: adv_mask == 0
    pct_range = range(0, 101, 10)
    eval_method = "occlusion"
    create_eval_dataset(conf, eval_method, occlusion_condition, pct_range)


def create_revelation_dataset(conf: utils.CONF):
    eval_method = "revelation"
    revelation_condition = lambda adv_mask: adv_mask == 1
    pct_range = range(0, 101, 10)
    create_eval_dataset(conf, eval_method, revelation_condition, pct_range)


def create_eval_dataset(conf: utils.CONF, eval_method: str, mask_condition, pct_range):

    model_name = "_".join(conf.MODEL.split("_")[:-2])
    # Paths for dataset and associated outputs
    BACKGROUND_DIR = get_dir_path(conf.DATASET_BACKGROUND_DIR, conf.DATASET_SPLIT)

    # TODO: improve naming (including/excluding train/test-split)
    DATASET_DIR = get_dir_path(conf.DATA_SET_PATH, conf.DATASET_SPLIT)
    XAI_DIR = get_dir_path(
        conf.XAIEV_BASE_DIR, "XAI_results", model_name, conf.XAI_METHOD, conf.DATASET_SPLIT
    )
    ADV_FOLDER = os.path.join(conf.EVAL_DATA_BASE_PATH, eval_method)

    CATEGORIES, image_dict = create_image_dict(conf.XAIEV_BASE_DIR, conf.DATASET_NAME, conf.DATASET_SPLIT)
    print(CATEGORIES)

    generate_adversarial_examples(
        adv_folder=ADV_FOLDER,
        pct_range=pct_range,
        categories=CATEGORIES,
        image_dict=image_dict,
        img_path=DATASET_DIR,
        background_dir=BACKGROUND_DIR,
        xai_dir=XAI_DIR,
        mask_condition=mask_condition,
        limit=conf.LIMIT,
    )
