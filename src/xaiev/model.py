## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


def get_model(model_name, n_classes):

    # drop extensions like "_1_1"
    model_name = get_short_model_name(model_name)

    if model_name == "resnet18":
        base_model = torchvision.models.resnet18()
        classifier_layer = torch.nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model = base_model
        model.fc = classifier_layer
        return model
    if model_name == "resnet50":
        base_model = torchvision.models.resnet50()
        classifier_layer = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        model = base_model
        model.fc = classifier_layer
        return model
    if model_name == "resnet152":
        base_model = torchvision.models.resnet18()
        classifier_layer = torch.nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model = base_model
        model.fc = classifier_layer
        return model
    if model_name == "resnext":
        base_model = torchvision.models.resnext50_32x4d()
        classifier_layer = torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        model = base_model
        model.fc = classifier_layer
        return model
    if model_name == "vgg16":
        base_model = torchvision.models.vgg16()
        classifier_layer = torch.nn.Linear(in_features=4096, out_features=n_classes, bias=True)
        model = base_model
        model.classifier[-1] = classifier_layer
        return model

    if model_name == "vgg16_dc":
        base_model = torchvision.models.vgg16()
        classifier_layer = torch.nn.Linear(in_features=4096, out_features=n_classes, bias=True)

        model = base_model
        layer_dc_1 = DWConfBlock(512)
        model.features[26] = layer_dc_1
        # model.features[7] = layer_dc_2
        model.classifier[-1] = classifier_layer
        return model

    if model_name == "alexnet":
        base_model = torchvision.models.alexnet()
        classifier_layer = torch.nn.Linear(in_features=4096, out_features=n_classes, bias=True)
        model = base_model
        model.classifier[-1] = classifier_layer
        return model
    if model_name == "densenet":
        base_model = torchvision.models.densenet121()
        classifier_layer = torch.nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        model = base_model
        model.classifier = classifier_layer
        return model
    if model_name == "convnext_tiny":
        base_model = torchvision.models.convnext_tiny()
        classifier_layer = torch.nn.Linear(in_features=768, out_features=n_classes, bias=True)
        model = base_model
        model.classifier[-1] = classifier_layer
        return model
    if model_name == "efficientnet":
        base_model = torchvision.models.efficientnet_b1()
        classifier_layer = torch.nn.Linear(in_features=1280, out_features=n_classes, bias=True)
        model = base_model
        model.classifier[-1] = classifier_layer
        return model
    if model_name == "simple_cnn":
        return SimpleCNN(n_classes)
    if model_name == "advanced_cnn":
        return ImprovedCNN(n_classes)


def get_short_model_name(model_full_name):
    """
    Convert "simple_cnn_1_1" to "simple_cnn" etc.
    """
    import re

    re_match = re.match("(.*?)_[_\d]+", model_full_name)
    if re_match:
        return re_match.groups(1)[0]
    else:
        return model_full_name


def load_model(model, optimizer, scheduler, filepath, device):
    # cpt = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    cpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(cpt["model"])
    optimizer.load_state_dict(cpt["optimizer"])
    scheduler.load_state_dict(cpt["scheduler"])
    return cpt["epoch"], cpt["trainstats"]


def test_model(model, testloader, criterion, device, debug_prints=False):
    model.eval()
    num_classes = len(testloader.dataset.get_classes())
    correct = torch.zeros(num_classes, dtype=torch.int64, device=device)
    correct_top5 = torch.zeros(num_classes, dtype=torch.int64, device=device)
    total = torch.zeros(num_classes, dtype=torch.int64, device=device)
    softmaxes = torch.zeros(num_classes, dtype=torch.float, device=device)
    scores = torch.zeros(num_classes, dtype=torch.float, device=device)
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)  # Move images and labels to the GPU
            outputs = model(images)
            softmax = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)  # Calculate the loss
            test_loss += loss.item()  # Accumulate the loss

            _, predicted = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, 1)

            for i in range(len(predicted)):
                correct[labels[i]] += predicted[i] == labels[i]
                correct_top5[labels[i]] += labels[i] in predicted_top5[i]
                total[labels[i]] += 1
                softmaxes[labels[i]] += softmax[i][labels[i]]
                scores[labels[i]] += outputs[i][labels[i]]

    accuracy_per_class = correct.float() / total.float()
    top5_accuracy_per_class = correct_top5.float() / total.float()
    test_loss /= len(testloader)  # Calculate the average test loss

    if debug_prints:
        print(f"Test Total Accuracy: {accuracy_per_class.mean():.2%}")
        print(f"Test Total Top-5 Accuracy: {top5_accuracy_per_class.mean():.2%}")

        for i in range(num_classes):
            print(f"Class {i} Test Accuracy: {accuracy_per_class[i]:.2%}")
            print(f"Class {i} Test Top-5 Accuracy: {top5_accuracy_per_class[i]:.2%}")

    model.train()  # Set the model back to training mode

    return (
        correct.cpu().numpy(),
        correct_top5.cpu().numpy(),
        total.cpu().numpy(),
        test_loss,
        softmaxes.cpu().numpy(),
        scores.cpu().numpy(),
    )


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.stem = nn.Conv2d(3, 32, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.stem(x)))
        x = self.bn2(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn3(F.relu(self.conv2(x)))
        x = self.bn4(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class ImprovedCNN(nn.Module):
    def __init__(self, n_classes):
        super(ImprovedCNN, self).__init__()
        self.stem = nn.Conv2d(3, 32, 7, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.stem(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn2(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.bn3(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.dropout1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class DWConfBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.pwconv1 = nn.Conv2d(
            dim, dim, kernel_size=1
        )  # pointwise/1x1 convs, implemented with linear layers

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        return x
