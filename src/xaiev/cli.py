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
        choices=["train", "inference", "create-saliency-maps", "create-eval-images", "eval", "visualize"],
        help="main xaiev command",
        nargs="?",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Full model name (e.g., simple_cnn_1_1)",
    )

    parser.add_argument(
        "--xai-method",
        type=str,
        help="specify the XAI method (e.g. gradcam, xrai, prism, lime)",
    )

    parser.add_argument("--version", action="store_true", help="print current version and exit")

    parser.add_argument(
        "--dataset_split", type=str, default="test", help="Dataset split (e.g., 'train', 'test')."
    )

    parser.add_argument("--random_seed", type=int, default=1414, help="Random seed for reproducibility.")

    parser.add_argument(
        "--bootstrap", action="store_true", help="create .env configuration file in current workdir"
    )

    parser.add_argument(
        "--inference", action="store_true", help="apply selected model to dataset to perform classification"
    )

    parser.add_argument("--inference-mode", "-im", choices=["copy", "json"], default="copy")

    parser.add_argument("--eval-method", choices=["revelation", "occlusion"], default="revelation")

    parser.add_argument(
        "--limit",
        type=int,
        help="limits the number of processed images (per class) to achieve faster testing",
    )

    parser.add_argument("--debug", action="store_true", help="start interactive debug mode; then exit")

    args = parser.parse_args()

    if args.bootstrap:
        core.bootstrap()
        exit()

    if args.version:
        from .release import __version__

        print(__version__)
        exit()

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
        if args.xai_method == "int_g":
            core.do_int_g_pipeline(args.model, CONF)
        elif args.xai_method == "lime":
            core.do_lime_pipeline(args.model, CONF)
        elif args.xai_method == "xrai":
            core.do_xrai_pipeline(args.model, CONF)
        elif args.xai_method == "prism":
            core.do_prism_pipeline(args.model, CONF)
        elif args.xai_method == "gradcam":
            # TODO: improve function name
            core.do_gradcam_pipeline(args.model, CONF)
        else:
            raise NotImplementedError(msg)

    elif args.command == "create-eval-images":
        utils.ensure_xai_method_and_model(args)
        core.create_eval_images(CONF)

    elif args.command == "eval":
        utils.ensure_xai_method_and_model(args)
        core.do_evaluation(CONF)

    elif args.command == "visualize":
        core.do_visualization(CONF)
