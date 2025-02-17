from . import utils
from .utilmethods import get_dir_path, create_image_dict, generate_adversarial_examples
from ipydex import IPS


def create_revelation_dataset(conf: utils.CONF, xai_method: str, model: str):

    # abbreviation

    # Paths for dataset and associated outputs
    GROUND_TRUTH_DIR = get_dir_path(conf.XAIEV_BASE_DIR, f"{conf.DATASET_NAME}_mask", conf.DATASET_SPLIT)
    BACKGROUND_DIR = get_dir_path(conf.XAIEV_BASE_DIR, f"{conf.DATASET_NAME}_background", conf.DATASET_SPLIT)
    DATASET_DIR = get_dir_path(conf.XAIEV_BASE_DIR, conf.DATASET_NAME, conf.DATASET_SPLIT)
    XAI_DIR = get_dir_path(conf.XAIEV_BASE_DIR, "XAI_results", model, xai_method, conf.DATASET_SPLIT)
    ADV_FOLDER = get_dir_path(
        conf.XAIEV_BASE_DIR, "XAI_evaluation", model, xai_method, "revelation", check_exists=False
    )


    CATEGORIES, image_dict = create_image_dict(conf.XAIEV_BASE_DIR, conf.DATASET_NAME, conf.DATASET_SPLIT)
    print(CATEGORIES)

    revelation_condition = lambda adv_mask: adv_mask == 1

    generate_adversarial_examples(
        adv_folder=ADV_FOLDER,
        pct_range=range(0, 101, 10),
        categories=CATEGORIES,
        image_dict=image_dict,
        #img_path=IMAGES_PATH,
        img_path=DATASET_DIR,
        background_dir=BACKGROUND_DIR,
        xai_dir=XAI_DIR,
        # this
        mask_condition=revelation_condition
    )

    IPS()
