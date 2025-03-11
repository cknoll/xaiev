import os

from . import utils

# note: some imports are done in the functions below, to achieve faster execution for individual commands


def bootstrap():
    """
    Function to conveniently create the .env file
    """

    fpath = ".env"
    if os.path.exists(fpath):
        msg = f"File {fpath} already exists. Nothing changed."
        print(utils.yellow(msg))
        exit(1)

    cwd = os.path.abspath(os.getcwd()).replace("\\", "/")

    print(f"Create .env file in current working directory: {cwd}.")
    content = (
        "# Note: This directory might contain several GB of (auto-generated) data\n"
        f'XAIEV_BASE_DIR="{cwd}"\n'
    )

    with open(fpath, "w") as fp:
        fp.write(content)
    print("\nDone.", "Please edit this file now and check for the correct data path (see README.md).")


def do_inference(*args, **kwargs):
    from . import inference

    inference.main(*args, **kwargs)


def do_gradcam_pipeline(*args, **kwargs):
    from . import gradcamheatmap

    gradcamheatmap.main(*args, **kwargs)


def do_int_g_pipeline(*args, **kwargs):
    from . import int_g_pipeline

    int_g_pipeline.main(*args, **kwargs)


def do_xrai_pipeline(*args, **kwargs):
    from . import Xrai_pipeline

    Xrai_pipeline.main(*args, **kwargs)


def do_lime_pipeline(*args, **kwargs):
    from . import lime_pipeline

    lime_pipeline.main(*args, **kwargs)


def do_prism_pipeline(*args, **kwargs):
    from . import PRISM_pipeline

    PRISM_pipeline.main(*args, **kwargs)


def create_eval_images(conf: utils.CONF):
    from . import eval_ds_creation

    eval_ds_creation.create_revelation_dataset(conf)
    eval_ds_creation.create_occlusion_dataset(conf)


def do_evaluation(conf: utils.CONF):
    from . import evaluation

    if conf.EVAL_METHOD == "revelation":
        evaluation.eval_revelation(conf)
        evaluation.visualize_evaluation(conf)
    elif conf.EVAL_METHOD == "occlusion":
        evaluation.eval_occlusion(conf)
        evaluation.visualize_evaluation(conf)
    else:
        raise ValueError(f"unexpected evaluation method: {conf.EVAL_METHOD}")


def do_visualization(CONF):
    from . import visualization

    visualization.main(CONF)
