import os
import shutil
import glob
import json
import random

import tqdm
from PIL import Image
import torch
from torchvision import transforms

# Personal debug module (`pip install ipydex`)
from ipydex import IPS

from .model import get_model
from . import utils


# shortcut
pjoin = os.path.join


class InferenceManager:
    """
    Class to bundle inference functionality.

    There are two ways of providing data:
        - a) individual files in `inference/images_to_classify`
        - b) just use all files from `<dataset_name>/test/*/`

    There are two modes to provide results:
        - a) mode="copy" (default)
            - images are copied to `inference/classification_results/<model_name>/<class_name>/
        - b) mode="json"
            - a json file with all results is produced:
            - inference/classification_results/<model_name>/classification_results.json



    """

    def __init__(self, model_full_name: str, conf: utils.CONF):

        self.model_full_name = model_full_name
        self.inference_data_base_path = conf.INFERENCE_DATA_BASE_PATH
        self.dataset_base_path = conf.DATA_SET_PATH
        self.model_cp_base_path = conf.MODEL_CP_PATH
        self.mode = conf.INFERENCE_MODE
        self.output_dir_path = pjoin(
            self.inference_data_base_path, "classification_results", self.model_full_name
        )

        self.class_names = None
        self.set_class_names()

        # Transformation for inference images
        self.transform_inference = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = None
        self.load_model_and_weights()

    def run(self):
        input_folder = os.path.join(self.inference_data_base_path, "images_to_classify")
        if os.path.exists(input_folder):
            input_paths = glob.glob(pjoin(input_folder, "*"))
        else:
            new_path = pjoin(self.dataset_base_path, "test")
            msg = (
                f"Expected input directory {input_folder} does not exist. "
                f"Using test fraction of dataset instead: {new_path}"
            )
            print(utils.yellow(msg))
            input_paths = self.get_image_paths_of_dataset_part(part="test")

        if self.mode == "json":
            self.classify_with_json_result(input_paths)
        else:
            # this is the original mode
            # mode == "copy"

            # Organize images into class folders
            self.organize_images(input_paths)

    def load_model_and_weights(self):

        # Derive model path and model name
        model_fname = f"{self.model_full_name}.tar"
        model_fpath = pjoin(self.model_cp_base_path, model_fname)
        model_name = "_".join(self.model_full_name.split("_")[:-2])  # Extract model_name

        # load model architecture
        self.model = get_model(model_name=model_name, n_classes=len(self.class_names)).to(self.device)

        checkpoint = torch.load(
            model_fpath, map_location=self.device, weights_only=False
        )  # Load to CPU or GPU
        self.epoch = checkpoint["epoch"]
        self.trainstats = checkpoint["trainstats"]
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()  # Set model to evaluation mode
        print(f"Loaded model: {model_name} | Epoch: {self.epoch}")

    def predict_image(self, model, image_path, class_names, full_res=False):
        """
        Function to predict class for an image
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform_inference(image).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        if full_res:
            return {
                "outputs": to_list(outputs),
                "predicted": to_list(predicted),
                "class": class_names[predicted.item()],
            }
        else:
            return class_names[predicted.item()]

    def set_class_names(self, part: str = "test"):
        """
        :param part:    either "test" or "train"
        """
        class_dirs = glob.glob(pjoin(self.dataset_base_path, part, "*"))
        self.class_names = [os.path.split(abspath)[1] for abspath in class_dirs]

    def get_image_paths_of_dataset_part(self, part: str = "test"):

        class_dirs = glob.glob(pjoin(self.dataset_base_path, part, "*"))
        assert len(class_dirs) == len(self.class_names)

        all_paths = []

        for class_dir in class_dirs:
            image_paths_for_class = glob.glob(pjoin(class_dir, "*"))
            image_paths_for_class.sort()
            all_paths.extend(image_paths_for_class)

        return all_paths

    def organize_images(self, input_paths: list[str]):
        """
        organize images into class folders (copy mode)

        :param input_paths:     list of absolute paths of the images
        """

        # Clear the output folder at the beginning
        if os.path.exists(self.output_dir_path):
            shutil.rmtree(self.output_dir_path)  # Remove all contents in the output folder
        os.makedirs(self.output_dir_path, exist_ok=True)

        # Create class folders
        for class_name in self.class_names:
            class_folder = os.path.join(self.output_dir_path, class_name)
            os.makedirs(class_folder, exist_ok=True)

        # Process each image in the list
        for file_path in input_paths:
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                filename = os.path.split(file_path)[1]
                predicted_class = self.predict_image(self.model, file_path, self.class_names)
                dest_folder = os.path.join(self.output_dir_path, predicted_class)
                shutil.copy(file_path, os.path.join(dest_folder, filename))
                print(f"Copied {filename} to {dest_folder}")

    def classify_with_json_result(self, input_paths: list[str]):
        """

        :param input_paths:     list of absolute paths of the images
        """

        random.shuffle(input_paths)

        # make testing faster
        # input_paths = input_paths[:30]

        result_dict = {}

        # TODO: detect HPC in a different way
        if not "horse" in self.inference_data_base_path:
            input_paths = tqdm.tqdm(input_paths)

        for image_path in input_paths:
            res = self.predict_image(self.model, image_path, self.class_names, full_res=True)

            # example: image_path = "/home/username/xaiev/data/atsds_large/imgs_main/test/00002/000096.png'"
            short_path = os.path.join(*image_path.split(os.path.sep)[-3:])
            class_dir = image_path.split(os.path.sep)[-2]
            boolean_result = class_dir == res["class"]
            res["boolean_result"] = boolean_result
            result_dict[short_path] = res

        json_fpath = pjoin(self.output_dir_path, "classification_results.json")
        os.makedirs(self.output_dir_path, exist_ok=True)
        with open(json_fpath, "w") as fp:
            json.dump(result_dict, fp, indent=2)

        print(f"file written: {json_fpath}")


# end of class InferenceManager


def to_list(tensor):
    res = tensor.cpu().squeeze().tolist()
    if isinstance(res, list):
        res2 = [round(elt, 3) for elt in res]
    else:
        res2 = res
    return res2


def main(model_full_name, conf: utils.CONF):

    im = InferenceManager(model_full_name, conf)
    im.run()
