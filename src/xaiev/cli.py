import argparse

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

from . import core
from . import utils

def main():

    # useful link https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    # for every argument we also have a short form
    parser.add_argument(
        "--model-full-name",
        "--model_full_name",  # note: --model_full_name is accepted for legacy reasons only
        "-n",
        type=str,
        help="Full model name (e.g., simple_cnn_1_1)",
    )

    parser.add_argument(
        "--bootstrap", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--inference", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--inference-mode", "-im", choices=["copy", "json"], default="copy"
    )

    parser.add_argument('--dataset_name', type=str, default="atsds_large", help="Name of the dataset.")

    args = parser.parse_args()

    if args.bootstrap:
        core.bootstrap()
        return

    CONF = utils.create_config(args)

    if args.inference:
        if args.model_full_name is None:
            msg = "Command line argument `--model-full-name` missing. Cannot continue."
            print(utils.bred(msg))
            exit(1)

        core.do_inference(args.model_full_name, CONF)
