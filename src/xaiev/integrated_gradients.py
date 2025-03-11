import torch
import numpy as np


def _get_ig_attribution_baseline(model, image_tensor, label_idx, runs, baseline=None):
    if baseline == None:
        baseline = torch.zeros_like(image_tensor)

    grads = []
    for i in range(0, runs):
        alpha = float(i) / float(runs)
        current_image = baseline + (alpha * (image_tensor - baseline))
        current_image.requires_grad = True
        out = model(current_image.squeeze().unsqueeze(0))
        out[:, label_idx].backward(retain_graph=True)

        grads.append(current_image.grad.cpu().detach().numpy())
    grads = grads[1:-1]
    avg_grad = np.mean(grads, axis=0)
    attribs = (image_tensor.cpu().numpy() - baseline.cpu().numpy()) * avg_grad
    return attribs


def get_ig_attribution_baseline(model, image_tensor, label_idx, runs, baseline=None):
    if baseline == None:
        attr_b = _get_ig_attribution_baseline(
            model, image_tensor, label_idx, runs, baseline=torch.zeros_like(image_tensor)
        )
        attr_w = _get_ig_attribution_baseline(
            model, image_tensor, label_idx, runs, baseline=torch.ones_like(image_tensor)
        )
        return (attr_b + attr_w) / 2
    return _get_ig_attribution_baseline(model, image_tensor, label_idx, runs, baseline=baseline)


def get_ig_attributions(model, image_tensor, label_idx, runs, baseline=None):
    attrib_grads = get_ig_attribution_baseline(model, image_tensor, label_idx, runs, baseline)
    attribs = (image_tensor.cpu() * attrib_grads).squeeze().cpu().detach().numpy()
    return attribs


def get_vicinity_map(attributions, windowssize=20):
    hmap = np.zeros_like(attributions)
    for i in range(hmap.shape[0]):
        for j in range(hmap.shape[1]):
            for k in range(hmap.shape[2]):
                startwindow_x = max(j - windowssize, 0)
                endwindow_x = min(j + windowssize, hmap.shape[1] - 1)
                startwindow_y = max(k - windowssize, 0)
                endwindow_y = min(k + windowssize, hmap.shape[2] - 1)
                hmap[i][j][k] = np.mean(
                    attributions[i][startwindow_x:endwindow_x][:, startwindow_y:endwindow_y]
                )
    return hmap
