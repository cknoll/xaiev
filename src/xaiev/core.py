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

    print("Create .env file in local working directory.")
    content = (
        "# Note: This directory might contain several GB of (auto-generated) data\n"
        'XAIEV_BASE_DIR="/home/username/xaiev/data"\n'
    )

    with open(fpath, "w") as fp:
        fp.write(content)
    print("\nDone.", "Please edit this file now and add the correct data path (see README.md).")


def do_inference(*args, **kwargs):
    from . import inference
    inference.main(*args, **kwargs)
