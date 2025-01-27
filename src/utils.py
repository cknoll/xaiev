import os
import argparse
from dotenv import load_dotenv
from types import SimpleNamespace  # used as flexible Container Class

def get_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model_full_name", "-n", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)"
    )

    # those two parameters are taken from .env now
    # parser.add_argument("--model_cp_base_path", "-cp", type=str, help="directory of model checkpoints", default=None)
    # parser.add_argument("--data_base_path", "-d", type=str, help="data path", default=None)
    return parser



def read_conf_from_dotenv() -> SimpleNamespace:
    assert os.path.isfile(".env")
    load_dotenv()

    conf = SimpleNamespace()
    conf.BASE_DIR = os.getenv("BASE_DIR")

    assert conf.BASE_DIR is not None
    return conf
