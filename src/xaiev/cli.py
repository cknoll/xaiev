import argparse

from . import core
from . import utils

def main():

    # useful link https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model_full_name", "-n", type=str, help="Full model name (e.g., simple_cnn_1_1)",
    )

    parser.add_argument(
        "--bootstrap", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--inference", action="store_true", help="create .env configuration file in current workdir"
    )

    args = parser.parse_args()

    if args.bootstrap:
        core.bootstrap()
        return
    elif args.inference:
        if args.model_full_name is None:
            msg = "Command line argument `--model-full-name` missing. Cannot continue."
            print(utils.bred(msg))
            exit(1)
        core.inference()
