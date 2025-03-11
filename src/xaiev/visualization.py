import os
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from ipydex import IPS
import tabulate

from . import utils


pjoin = os.path.join


def main(conf: utils.CONF):
    em = EvaluationManager(conf)
    em.create_plots()
    em.calc_asq_values()


class EvaluationManager:

    def __init__(self, conf: utils.CONF):
        self.conf = conf

        # this is a temporary solution
        result_dir = pjoin(conf.XAIEV_BASE_DIR, "collected_results")

        self.files = glob.glob(pjoin(result_dir, "*.pcl"))
        self.files.sort()

        # TODO: load this dynamically from the available modules
        self.model_names = ["simple_cnn", "vgg16", "resnet50", "convnext_tiny"]
        self.xai_methods = ["gradcam", "lime", "prism", "xrai"]

        self.eval_methods = ["occlusion", "revelation"]

        # maps 3-tuples (model_name, xai_method, eval_method) to the accuracy curves
        self.curves = {}
        self.ASQ_values: list[list] = []

        # allow no-plot-mode for faster ASQ-calculation
        self.plot = True

    def create_plots(self):

        for model_name in self.model_names:
            for eval_method in self.eval_methods:
                self.create_plot_for_model_and_eval_method(model_name, eval_method)

    def calc_asq_values(self):

        acc_max_list = []
        acc_min_list = []

        for model_name in self.model_names:
            key_occ = (model_name, self.xai_methods[0], "occlusion")
            key_rev = (model_name, self.xai_methods[0], "revelation")

            # maximum accuracy for each model: 0% occlusion
            acc_max = self.curves[key_occ][1][0]
            acc_max_list.append(acc_max)
            # minimum accuracy for each model: 0% revelation
            acc_min = self.curves[key_rev][1][0]
            acc_min_list.append(acc_min)

            # assume the same percentage range for occ and rev (should be 0% ... 10%)
            # select first and last value

            percentage_range = np.array(self.curves[key_occ][0])
            pct_min, pct_max = percentage_range[[0, -1]]

            A_total = (acc_max - acc_min) * (pct_max - pct_min)

            self.ASQ_values.append([utils.get_model_display_name(model_name)])

            for xai_method in self.xai_methods:
                for eval_method in self.eval_methods:
                    key = (model_name, xai_method, eval_method)

                    _, accuracies = self.curves[key]

                    area_under_curve = np.trapezoid(accuracies, percentage_range) - (
                        acc_min * (pct_max - pct_min)
                    )
                    area_above_curve = A_total - area_under_curve

                    if eval_method == "occlusion":
                        A_oc_1 = area_above_curve
                        # A_oc_2 = area_under_curve
                    elif eval_method == "revelation":
                        A_re_1 = area_under_curve
                        # A_re_2 = area_above_curve

                ASQ = round(0.5 * (A_oc_1 + A_re_1) / A_total, 2)

                # add value to table
                self.ASQ_values[-1].append(ASQ)

        headers = [""] + [utils.get_xai_method_display_name(n) for n in self.xai_methods]
        print(tabulate.tabulate(self.ASQ_values, tablefmt="latex", headers=headers))

    def create_plot_for_model_and_eval_method(self, model_name, eval_method):

        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        mm = 1 / 25.4  # mm to inch
        scale = 1.75
        fs = [75 * mm * scale, 35 * mm * scale]
        fig = plt.figure(figsize=fs, dpi=100)

        for fpath in self.files:
            _, fname = os.path.split(fpath)

            if not fname.startswith(model_name):
                continue
            if eval_method not in fpath:
                continue
            # TODO: change this to fpath.split(os.path.sep)
            xai_method = fname.split("__")[1]
            self.eval_pcl_file(fpath, model_name, xai_method, eval_method)

        if self.plot:

            model_display_name = utils.get_model_display_name(model_name)
            plt.title(f"{model_display_name} ({eval_method})")
            img_fpath = pjoin(self.conf.XAIEV_BASE_DIR, f"img_{model_name}_{eval_method}.pdf")
            plt.xlim(-0.2, 11.8)
            plt.ylim(1, 105)
            plt.xlabel(r"$T$ [\%]")
            plt.ylabel(r"Accuracy [\%]")
            plt.legend(bbox_to_anchor=(0.85, 0.8), loc="upper left")
            plt.subplots_adjust(bottom=0.22, left=0.12, right=0.87)
            # if model_name == "simple_cnn" and eval_method == "revelation":
            #     plt.show()
            #     exit()

            plt.savefig(img_fpath)
            print(f"File written: {img_fpath}")

    def eval_pcl_file(self, fpath, model_name, xai_method, eval_method, label=None, plt_args=None):
        with open(fpath, "rb") as f:
            performance_xai_type = pickle.load(f)

        if label is None:
            label = utils.get_xai_method_display_name(xai_method)
        if plt_args is None:
            plt_args = {}

        correct, correct_5, softmax, score, loss = performance_xai_type[xai_method]

        # calculate accuracy in percent
        accuracy_pct = np.mean((np.divide(correct, 50)), axis=1) * 100

        # TODO: remove this when the complete workflow includes 0% for occlusion

        # the following values come from a tailored run (only 0% occlusion for prims)
        # motivation save time but get the correct values for the paper

        zero_values = {
            "simple_cnn": 98.94736842,
            "vgg16": 99.26315789,
            "resnet50": 98.63157895,
            "convnext_tiny": 97.68421053,
        }

        if np.diff(accuracy_pct[[0, -1]]) < 0:
            accuracy_pct = np.concatenate(([zero_values[model_name]], accuracy_pct))

        percentage_range = list(range(len(accuracy_pct)))

        key = (model_name, xai_method, eval_method)
        self.curves[key] = (percentage_range, accuracy_pct)

        if self.plot:
            plt.plot(percentage_range, accuracy_pct, label=label, **plt_args)

    def calculate_areas(self, xx, yy):
        pass
