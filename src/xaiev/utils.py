import os

from dotenv import load_dotenv
from colorama import Style, Fore

# Writeable container which will store the config

from dataclasses import dataclass

@dataclass
class CONF:
    XAIEV_BASE_DIR: str
    DATA_SET_PATH: str
    DATASET_NAME: str
    DATASET_BACKGROUND_DIR: str
    DATASET_MASK: str
    MODEL_CP_PATH: str
    INFERENCE_DATA_BASE_PATH: str
    INFERENCE_MODE: str
    DATASET_NAME: str
    DATASET_SPLIT: str
    RANDOM_SEED: int
    LIMIT: int
    MODEL: str
    XAI_METHOD: str

def read_conf_from_dotenv() -> CONF:
    if not os.path.isfile(".env"):
        msg = "Could not find configuration file (.env). Please see section 'Bootstrapping' in README.md."
        raise FileNotFoundError(msg)
    load_dotenv()

    CONF.XAIEV_BASE_DIR = os.getenv("XAIEV_BASE_DIR")

    assert CONF.XAIEV_BASE_DIR is not None
    return CONF


def create_config(args) -> CONF:
    read_conf_from_dotenv()  # manipulate global variable CONF
    CONF.DATA_SET_PATH = os.path.join(CONF.XAIEV_BASE_DIR, "dataset_main")

    # the following names are now hardcoded (according to directory structure specified in README)
    CONF.DATASET_NAME = "dataset_main"
    CONF.DATASET_BACKGROUND_DIR = os.path.join(CONF.XAIEV_BASE_DIR, "dataset_background")
    CONF.DATASET_MASK = os.path.join(CONF.XAIEV_BASE_DIR, "dataset_mask")

    CONF.MODEL_CP_PATH = os.path.join(CONF.XAIEV_BASE_DIR, "model_checkpoints")
    CONF.INFERENCE_DATA_BASE_PATH = os.path.join(CONF.XAIEV_BASE_DIR, "inference")
    CONF.INFERENCE_MODE = args.inference_mode
    CONF.DATASET_SPLIT = args.dataset_split
    CONF.RANDOM_SEED = args.random_seed
    CONF.LIMIT = args.limit
    CONF.MODEL = args.model
    CONF.XAI_METHOD = args.xai_method

    return CONF


def ensure_xai_method(args):
    if args.xai_method is None:
        msg = "Command line argument `--xai-method` missing. Cannot continue."
        print(bred(msg))
        exit(1)


def ensure_model(args):
    if args.model is None:
        msg = "Command line argument `--model` missing. Cannot continue."
        print(bred(msg))
        exit(1)


def ensure_xai_method_and_model(args):
    ensure_xai_method(args)
    ensure_model(args)


################################################################################
# functions to create colored console outputs
################################################################################

def bright(txt):
    return f"{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def bgreen(txt):
    return f"{Fore.GREEN}{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def bred(txt):
    return f"{Fore.RED}{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def yellow(txt):
    return f"{Fore.YELLOW}{txt}{Style.RESET_ALL}"
