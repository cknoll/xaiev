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

    cwd = os.path.abspath(os.getcwd())
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


def do_gradcam(*args, **kwargs):
    from . import gradcamheatmap
    gradcamheatmap.main(*args, **kwargs)
