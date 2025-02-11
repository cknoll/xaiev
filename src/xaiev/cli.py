import argparse

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

from . import core
from . import utils

def main():

    # useful link https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    # TODO: for every argument we also should have a short form
    parser.add_argument(
        "--model",
        "--model-full-name",
        "--model_full_name",  # note: --model_full_name etc is accepted for legacy reasons only
        "-n",  # obsolete (legacy)
        "-m",
        type=str,
        help="Full model name (e.g., simple_cnn_1_1)",
    )

    parser.add_argument(
        '--dataset_name', type=str, default="atsds_large", help="Name of the dataset."
    )

    parser.add_argument(
        '--dataset_split', type=str, default="test", help="Dataset split (e.g., 'train', 'test')."
    )

    parser.add_argument(
        '--random_seed', type=int, default=1414, help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--bootstrap", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--inference", action="store_true", help="apply selected model to dataset to perform classification"
    )

    parser.add_argument(
        "--inference-mode", "-im", choices=["copy", "json"], default="copy"
    )

    parser.add_argument(
        "--gradcam", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--create-xai-saliency-maps",
        "-csm",
        metavar="XAI_METHOD",
        type=str,
        help="choose an XAI method to create the saliency maps",
    )


    parser.add_argument(
        "--debug", action="store_true", help="start interactive debug mode; then exit"
    )

    args = parser.parse_args()

    if args.bootstrap:
        core.bootstrap()
        return

    CONF = utils.create_config(args)

    if args.debug:
        IPS()
        exit()

    if args.inference:
        if args.model_full_name is None:
            msg = "Command line argument `--model` missing. Cannot continue."
            print(utils.bred(msg))
            exit(1)

        core.do_inference(args.model_full_name, CONF)

    # TODO: use create-xai-saliency-maps
    elif args.gradcam:
        if args.model_full_name is None:
            msg = "Command line argument `--model-full-name` missing. Cannot continue."
            print(utils.bred(msg))
            exit(1)

        core.do_gradcam(args.model_full_name, CONF)

    elif args.foo:
        pass
