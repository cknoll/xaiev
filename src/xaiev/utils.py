import os

from dotenv import load_dotenv
from colorama import Style, Fore

# Writeable container which will store the config

from dataclasses import dataclass

@dataclass
class CONF:
    XAIEV_BASE_DIR: str
    DATA_SET_PATH: str
    MODEL_CP_PATH: str
    INFERENCE_DATA_BASE_PATH: str
    INFERENCE_MODE: str
    DATASET_NAME: str
    DATASET_SPLIT: str
    RANDOM_SEED : int

def read_conf_from_dotenv() -> CONF:
    assert os.path.isfile(".env")
    load_dotenv()

    CONF.XAIEV_BASE_DIR = os.getenv("XAIEV_BASE_DIR")

    assert CONF.XAIEV_BASE_DIR is not None
    return CONF


def create_config(args) -> CONF:
    read_conf_from_dotenv()  # manipulate global variable CONF
    CONF.DATA_SET_PATH = os.path.join(CONF.XAIEV_BASE_DIR, args.dataset_name)
    CONF.MODEL_CP_PATH = os.path.join(CONF.XAIEV_BASE_DIR, "model_checkpoints")
    CONF.INFERENCE_DATA_BASE_PATH = os.path.join(CONF.XAIEV_BASE_DIR, "inference")
    CONF.INFERENCE_MODE = args.inference_mode
    CONF.DATASET_NAME = args.dataset_name
    CONF.DATASET_SPLIT = args.dataset_split
    CONF.RANDOM_SEED = args.random_seed

    return CONF


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