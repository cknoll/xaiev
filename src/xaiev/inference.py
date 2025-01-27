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


# shortcut
pjoin = os.path.join


class InferenceManager:
    """
    Class to bundle inference functionality.
    """

    def __init__(self, model_full_name: str, data_base_path: str, model_cp_base_path: str, mode: str):

        self.model_full_name = model_full_name
        self.data_base_path = data_base_path
        self.model_cp_base_path = model_cp_base_path
        self.mode = mode

        # use simple enumeration here (might be overwritten with directory names later)
        self.class_names = [f"{i:05d}" for i in range(1, 20)]  # e.g., 00001 to 00019

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
        if self.mode == "json":
            self.classify_with_json_result()
        else:
            # this is the original mode
            # mode == "copy"

            input_folder = os.path.join(self.data_base_path, "inference/images_to_classify")
            output_folder = os.path.join(self.data_base_path, "inference/classified_images")

            # Organize images into class folders
            self.organize_images(input_folder, output_folder)

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

    def get_image_paths(self, sub_path="test"):
        """
        return a list of absolute paths of all images e.g. from train/ or test/

        also: set self.class_names to appropriate directory names
        """

        host_path = pjoin(self.data_base_path, sub_path)
        class_dirs = glob.glob(pjoin(host_path, "*"))
        assert len(class_dirs) == len(self.class_names)
        dir_name_start_idx = len(host_path) + 1
        self.class_names = [path[dir_name_start_idx:] for path in class_dirs]

        all_paths = []

        for class_dir in class_dirs:
            image_paths_for_class = glob.glob(pjoin(class_dir, "*"))
            image_paths_for_class.sort()
            all_paths.extend(image_paths_for_class)

        return all_paths

    def organize_images(self, input_folder, output_folder):
        """
        organize images into class folders (copy mode)
        """
        # Clear the output folder at the beginning
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)  # Remove all contents in the output folder
        os.makedirs(output_folder, exist_ok=True)

        # Create class folders
        for class_name in self.class_names:
            class_folder = os.path.join(output_folder, class_name)
            os.makedirs(class_folder, exist_ok=True)

        # Process each image in the input folder
        for filename in os.listdir(input_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(input_folder, filename)
                predicted_class = self.predict_image(self.model, file_path, self.class_names)
                dest_folder = os.path.join(output_folder, predicted_class)
                shutil.copy(file_path, os.path.join(dest_folder, filename))
                print(f"Copied {filename} to {dest_folder}")

    def classify_with_json_result(self):

        all_img_paths = self.get_image_paths()

        random.shuffle(all_img_paths)

        # make testing faster
        all_img_paths = all_img_paths[:30]

        result_dict = {}
        path_start_idx = len(self.data_base_path) + 1

        if not "horse" in self.data_base_path:
            all_img_paths = tqdm.tqdm(all_img_paths)

        for image_path in all_img_paths:
            res = self.predict_image(self.model, image_path, self.class_names, full_res=True)

            # short_path = "test/00001/000000.png"
            short_path = image_path[path_start_idx:]
            train_test_dir, class_dir, fname = short_path.split(os.path.sep)
            boolean_result = class_dir == res["class"]
            res["boolean_result"] = boolean_result
            result_dict[short_path] = res

        json_fpath = "results.json"
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


def main(model_full_name, data_base_path=None, model_cp_base_path=None, mode="copy"):


    im = InferenceManager(
        model_full_name=model_full_name,
        data_base_path=data_base_path,
        model_cp_base_path=model_cp_base_path,
        mode=mode,
    )
    im.run()


if __name__ == "__main__":

    parser = utils.get_default_arg_parser()
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help=(
            "mode: 'copy' (default) or 'json'. Mode 'json' means a) the files are read from "
            "their original class-subdirs and b) the result is `results.json` and not a directory full of files"
        ),
        default="copy",
    )
    args = parser.parse_args()

    main(
        model_full_name=args.model_full_name,
        data_base_path=args.data_base_path,
        model_cp_base_path=args.model_cp_base_path,
        mode=args.mode,
    )
