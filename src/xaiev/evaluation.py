import os
import json
import math
import random
import pickle

# 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
from ipydex import IPS

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CUDA_LAUNCH_BLOCKING = 1

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import transforms as transforms

# our own modules
from .ATSDS import ATSDS
from .gradcam import get_gradcam
from .model import get_model, load_model, test_model
from . import utils


# TODO: unify common parts of both functions
def eval_revelation(conf: utils.CONF):
    _evaluation(conf, range(0, 101, 10))


def eval_occlusion(conf: utils.CONF):
    _evaluation(conf, range(0, 101, 10))


def _evaluation(conf: utils.CONF, percentage_range: list[int]):
    """
    Main functionality copied from the original notebooks
    """

    # Define transformations for the train and test dataset
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    RANDOM_SEED = conf.RANDOM_SEED

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Used for reproducibility to fix randomness in some GPU calculations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    # load dataset for fixed percentage to get number of classes
    # conf.EVAL_DATA_PATH is the directory which contains the percentage subdirs
    # (which contain the class-dirs which contain the images)

    data_set_path = os.path.join(conf.EVAL_DATA_PATH, "10")
    testset = ATSDS(root=data_set_path, split=None, dataset_type=None, transform=transform_test)

    model = get_model(conf.MODEL, n_classes=testset.get_num_classes())
    model = model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    # load the model weights
    epoch, trainstats = load_model(model, optimizer, scheduler, conf.MODEL_PATH, device)

    # might be useful for debugging/reporting
    # train_loss = trainstats[0]
    # test_loss = trainstats[1]
    # train_stats = trainstats[2]

    # xai_methods = ["gradcam","ig_fixpoints","lime","prism","xrai"]
    xai_methods = [conf.XAI_METHOD]

    performance_xai_type = {}

    # TODO: drop outer loop
    for current_method in xai_methods:
        c_list = []
        c_5_list = []
        softmaxes_list = []
        scores_list = []
        losses = []

        # Iterate over percentages
        for pct in percentage_range:
            data_set_path = os.path.join(conf.EVAL_DATA_PATH, str(pct))
            testset = ATSDS(root=data_set_path, split=None, dataset_type=None, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
            c, c_5, t, loss, softmaxes, scores = test_model(model, testloader, loss_criterion, device)
            c_list.append(c)
            c_5_list.append(c_5)
            softmaxes_list.append(softmaxes)
            scores_list.append(scores)
        performance_xai_type[current_method] = (c_list, c_5_list, softmaxes_list, scores_list, losses)

    total = t

    # Save the performance_xai_type dictionary to a pickle file
    dic_save_path = conf.EVAL_RESULT_DATA_PATH
    with open(dic_save_path, "wb") as f:
        pickle.dump(performance_xai_type, f)


def visualize_evaluation(conf: utils.CONF, xai_methods: list[str] = None):

    # Load the performance_xai_type dictionary from the pickle file
    dic_load_path = conf.EVAL_RESULT_DATA_PATH
    with open(dic_load_path, "rb") as f:
        performance_xai_type = pickle.load(f)

    if xai_methods is None:
        xai_methods = [conf.XAI_METHOD]

    accuracies = []
    for current_method in xai_methods:
        correct, correct_5, softmax, score, loss = performance_xai_type[current_method]
        accuracy = np.mean((np.divide(correct, 50)), axis=1)
        accuracies.append(accuracy)

    for i, entry in enumerate(accuracies):
        plt.plot(entry)
        plt.legend(xai_methods)
        plt.savefig(dic_load_path.replace(".pcl", ".png"))
