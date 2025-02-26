import os
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ipydex import IPS

from . import utils

pjoin = os.path.join

def main(conf: utils.CONF):
    # this is a temporary solution
    result_dir = pjoin(conf.XAIEV_BASE_DIR, "collected_results")

    files = glob.glob(pjoin(result_dir, "*.pcl"))
    files.sort()

    for fpath in files:
        _, fname = os.path.split(fpath)
        model_name = "resnet"
        if not fname.startswith(model_name):
            continue
        # TODO: change this to os.path.sep
        xai_method = fname.split("__")[1]
        plot_pcl_file(fpath, xai_method)
    plt.legend()
    plt.savefig(f"_img_{model_name}.png")

    IPS()


def plot_pcl_file(fpath, xai_method, label=None, plt_args=None):
    with open(fpath, 'rb') as f:
        performance_xai_type = pickle.load(f)

    if label is None:
        label = xai_method
    if plt_args is None:
        plt_args = {}

    accuracies = []
    correct, correct_5, softmax, score, loss = performance_xai_type[xai_method]
    accuracy = np.mean((np.divide(correct, 50)), axis=1)
    accuracies.append(accuracy)

    for i, entry in enumerate(accuracies):
        plt.plot(entry, label=label, **plt_args)
        # plt.legend(xai_methods)
        # plt.savefig(dic_load_path.replace(".pcl", ".png"))
