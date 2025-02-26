import os
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ipydex import IPS

from . import utils

pjoin = os.path.join

def main(conf: utils.CONF):
    em = EvaluationManager(conf)
    em.create_plots()

class EvaluationManager:

    def __init__(self, conf: utils.CONF):
        self.conf = conf

        # this is a temporary solution
        result_dir = pjoin(conf.XAIEV_BASE_DIR, "collected_results")

        self.files = glob.glob(pjoin(result_dir, "*.pcl"))
        self.files.sort()

    def create_plots(self):

        model_names = ["simple_cnn", "vgg16", "resnet50", "convnext_tiny"]
        eval_methods = ["occlusion", "revelation"]

        for model_name in model_names:
            for eval_method in eval_methods:
                self.create_plot_for_model_and_eval_method(model_name, eval_method)

    def create_plot_for_model_and_eval_method(self, model_name, eval_method):

        mm = 1/25.4 # mm to inch
        scale = 2
        fs = [75*mm*scale, 35*mm*scale]
        plt.figure(figsize=fs, dpi=100)
        plt.figure(figsize=fs)
        for fpath in self.files:
            _, fname = os.path.split(fpath)

            if not fname.startswith(model_name):
                continue
            if eval_method not in fpath:
                continue
            # TODO: change this to fpath.split(os.path.sep)
            xai_method = fname.split("__")[1]
            plot_pcl_file(fpath, xai_method)
        plt.legend()
        plt.title(f"{model_name} {eval_method}")
        img_fpath = pjoin(self.conf.XAIEV_BASE_DIR, f"img_{model_name}_{eval_method}.png")
        plt.savefig(img_fpath)
        print(f"File written: {img_fpath}")


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
