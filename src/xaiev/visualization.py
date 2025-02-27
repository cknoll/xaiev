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
        model_display_names = {}
        eval_methods = ["occlusion", "revelation"]

        for model_name in model_names:
            for eval_method in eval_methods:
                self.create_plot_for_model_and_eval_method(model_name, eval_method)

    def create_plot_for_model_and_eval_method(self, model_name, eval_method):

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        mm = 1/25.4 # mm to inch
        scale = 1.75
        fs = [75*mm*scale, 35*mm*scale]
        fig = plt.figure(figsize=fs, dpi=100)


        for fpath in self.files:
            _, fname = os.path.split(fpath)

            if not fname.startswith(model_name):
                continue
            if eval_method not in fpath:
                continue
            # TODO: change this to fpath.split(os.path.sep)
            xai_method = fname.split("__")[1]
            plot_pcl_file(fpath, xai_method)

        model_display_name = utils.get_model_display_name(model_name)
        plt.title(f"{model_display_name} ({eval_method})")
        img_fpath = pjoin(self.conf.XAIEV_BASE_DIR, f"img_{model_name}_{eval_method}.pdf")
        plt.xlim(-0.2, 11.8)
        plt.ylim(1, 105)
        plt.xlabel(r"$T$ [\%]")
        plt.ylabel(r"Accuracy [\%]")
        plt.legend(bbox_to_anchor=(.85, 0.8), loc="upper left")
        plt.subplots_adjust(bottom=0.22, left=0.12, right=0.87)
        # if model_name == "simple_cnn" and eval_method == "revelation":
        #     plt.show()
        #     exit()

        plt.savefig(img_fpath)
        print(f"File written: {img_fpath}")


def plot_pcl_file(fpath, xai_method, label=None, plt_args=None):
    with open(fpath, 'rb') as f:
        performance_xai_type = pickle.load(f)

    if label is None:
        label = utils.get_xai_method_display_name(xai_method)
    if plt_args is None:
        plt_args = {}

    accuracies = []
    correct, correct_5, softmax, score, loss = performance_xai_type[xai_method]

    # calculate accuracy in percent
    accuracy_pct = np.mean((np.divide(correct, 50)), axis=1) * 100
    plt.plot(accuracy_pct, label=label, **plt_args)
