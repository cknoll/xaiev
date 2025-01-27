import os
from types import SimpleNamespace  # used as flexible Container Class

from dotenv import load_dotenv
from colorama import Style, Fore


def read_conf_from_dotenv() -> SimpleNamespace:
    assert os.path.isfile(".env")
    load_dotenv()

    conf = SimpleNamespace()
    conf.BASE_DIR = os.getenv("BASE_DIR")

    assert conf.BASE_DIR is not None
    return conf

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