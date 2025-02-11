import argparse

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

from . import core
from . import utils

def main():

    # useful link https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["train", "inference", "create-saliency-maps", "create-eval-images", "eval"],
        help="main xaiev command",
        nargs="?",
    )

    parser.add_argument(
        "--model",
        "--model-full-name",
        "--model_full_name",  # note: --model_full_name etc is accepted for legacy reasons only
        "-n",  # obsolete (legacy)
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

    if args.command == "train":
        utils.ensure_model(args)
        msg = "not yet implemented"
        raise NotImplementedError(msg)

    elif args.command == "inference":
        utils.ensure_model(args)
        core.do_inference(args.model, CONF)

    elif args.command == "create-saliency-maps":
        utils.ensure_xai_method_and_model(args)
        if args.xai_method != "gradcam":
            raise NotImplementedError()

        # TODO: improve function name
        core.do_gradcam(args.model, CONF)
    elif args.command == "create-eval-images":
        utils.ensure_xai_method_and_model(args)

    elif args.command == "eval":
        utils.ensure_xai_method_and_model(args)
