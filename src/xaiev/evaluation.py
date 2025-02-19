## Standard libraries
import os
import json
import math
import random
import pickle

# 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1

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


def main_occlusion(conf: utils.CONF):
    """
    Main functionality copied from the original notebook
    """

    # Define transformations for the train and test dataset
    transform_train = transforms.Compose(
        [transforms.Resize(256),
        transforms.RandomCrop(size=(224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_test = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Define paths using os.path.join for better cross-platform compatibility
    DATASET_PATH = os.path.join("data", "auswertung_hpc", "auswertung")
    MODEL_NAME = conf.MODEL
    XAI_NAME = conf.XAI_METHOD
    ADV_PCT = "10"
    dataset_type = os.path.join("occlusion", "10")
    dataset_split = conf.DATASET_SPLIT
    ROOT_PATH = os.path.join(DATASET_PATH, MODEL_NAME, XAI_NAME)

    RANDOM_SEED = conf.RANDOM_SEED

    # ---

    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Used for reproducibility to fix randomness in some GPU calculations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    testset = ATSDS(root=ROOT_PATH, split=dataset_split, dataset_type=dataset_type, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True, num_workers = 2)

    # ---

    model = get_model(conf.MODEL, n_classes=testset.get_num_classes())
    model = model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,200)

    epoch, trainstats = load_model(model, optimizer, scheduler, conf.MODEL_PATH, device)
    train_loss = trainstats[0]
    test_loss = trainstats[1]
    train_stats= trainstats[2]

    # xai_methods = ["gradcam","ig_fixpoints","lime","prism","xrai"]
    xai_methods = [conf.EVAL_METHOD]

    performance_xai_type = {}

    # ---

    # TODO: drop outer loop
    for current_method in xai_methods:
        ROOT_PATH = os.path.join(DATASET_PATH, MODEL_NAME, current_method)
        c_list = []
        c_5_list = []
        softmaxes_list = []
        scores_list = []
        losses = []

        # Iterate over occlusion percentages
        for pct in range(10,101,10):
            dataset_type = os.path.join("occlusion", str(pct))
            testset = ATSDS(root=ROOT_PATH, split=dataset_split, dataset_type=dataset_type, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True, num_workers = 2)
            c,c_5,t,loss,softmaxes,scores = test_model(model,testloader,loss_criterion, device)
            c_list.append(c)
            c_5_list.append(c_5)
            softmaxes_list.append(softmaxes)
            scores_list.append(scores)
        performance_xai_type[current_method] = (c_list,c_5_list,softmaxes_list,scores_list,losses)

    total = t

    # ---

    # Save the performance_xai_type dictionary to a pickle file
    dic_save_path = conf.EVAL_RESULT_DATA_PATH
    with open(dic_save_path, 'wb') as f:
        pickle.dump(performance_xai_type, f)


def visualize_occlusion(conf: utils.CONF, xai_methods: list[str]):

    # Load the performance_xai_type dictionary from the pickle file
    dic_load_path = conf.EVAL_RESULT_DATA_PATH
    with open(dic_load_path, 'rb') as f:
        performance_xai_type = pickle.load(f)

    accuracies = []
    for current_method in xai_methods:
        correct,correct_5,softmax,score,loss = performance_xai_type[current_method]
        accuracy = np.mean((np.divide(correct,50)),axis=1)
        accuracies.append(accuracy)

    for i, entry in enumerate(accuracies):
        plt.plot(entry)
        plt.legend(xai_methods)
