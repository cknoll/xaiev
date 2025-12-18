# This sctipt is not yet fully tested. Please run with caution.


import os
from PIL import Image
import numpy as np


# for i in range(19):
#     background_files = os.listdir("atsds_large/imgs_background/train")
#     for file in background_files:
#         background_images_file = os.path.join("atsds_large/imgs_background/train", file)
#         for background_imgs_path in os.listdir(background_images_file):
#             background_image = np.array(Image.open(os.path.join("atsds_large/imgs_background/train", file, background_imgs_path)))
#             for train_files in os.listdir("atsds_large/imgs_main/train"):
#                 train_image = np.array(Image.open(os.path.join("atsds_large/imgs_main/train", train_files, background_imgs_path)))
#                 mask        = np.array(Image.open(os.path.join("atsds_large/imgs_mask/train", train_files, background_imgs_path)).convert("L"))
#                 mask = (mask > 0).astype(np.uint8)
#                 new_train_image = (background_image * (1 - mask[:, :, None]) + train_image * mask[:, :, None]).astype(np.uint8)
#                 os.makedirs(os.path.join("atsds_large_all_background/train", train_files), exist_ok=True)
#                 Image.fromarray(new_train_image).save(os.path.join("atsds_large_all_background/train", train_files, background_imgs_path))

# for train_imgs_folder in os.listdir("atsds_large/imgs_main/train"):
#     #os.makedirs(os.path.join("atsds_large_all_background/imgs_main_extend/train", train_imgs_folder), exist_ok=True)
#     train_imgs_path = sorted(os.listdir(os.path.join("atsds_large/imgs_main/train", train_imgs_folder)))
    
#     for j in range(450):
#         train_img = np.array(Image.open(os.path.join("atsds_large/imgs_main/train", train_imgs_folder, train_imgs_path[j])))
#         mask     = np.array(Image.open(os.path.join("atsds_large/imgs_mask/train", train_imgs_folder, train_imgs_path[j])).convert("L"))
        
#         mask = (mask > 0).astype(np.uint8)
#         for background_imgs_folder in [f"atsds_large/imgs_background/train/{i}" for i in os.listdir("atsds_large/imgs_background/train")]:
#             background_imgs_path = sorted(os.listdir(background_imgs_folder))
#             background_image = np.array(Image.open(os.path.join(background_imgs_folder, background_imgs_path[j])))
#             new_train_image = (background_image * (1 - mask[:, :, None]) + train_img * mask[:, :, None]).astype(np.uint8)
#             Image.fromarray(new_train_image).save(os.path.join("atsds_large_all_background/imgs_main_extend/train", os.path.basename(background_imgs_folder), background_imgs_path[j]))

split = "train"

for train_imgs_folder in os.listdir(os.path.join("atsds_large/imgs_main", split)):
    os.makedirs(os.path.join("atsds_large_all_background/imgs_main_extend", split, train_imgs_folder), exist_ok=True)

for background_imgs_folder in os.listdir(os.path.join("atsds_large/imgs_background", split)):
    background_imgs_path = sorted(os.listdir(os.path.join("atsds_large/imgs_background", split, background_imgs_folder)))
    for idx, background_imgs_path_sorted in enumerate(background_imgs_path):
        background_image = np.array(Image.open(os.path.join("atsds_large/imgs_background", split, background_imgs_folder, background_imgs_path_sorted)))
        for train_imgs_folder in os.listdir(os.path.join("atsds_large/imgs_main", split)):
            train_imgs_path = sorted(os.listdir(os.path.join("atsds_large/imgs_main", split, train_imgs_folder)))
            train_img = np.array(Image.open(os.path.join("atsds_large/imgs_main", split, train_imgs_folder, train_imgs_path[idx])))
            mask     = np.array(Image.open(os.path.join("atsds_large/imgs_mask", split, train_imgs_folder, train_imgs_path[idx])).convert("L"))
            mask = (mask > 0).astype(np.uint8)
            new_train_image = (background_image * (1 - mask[:, :, None]) + train_img * mask[:, :, None]).astype(np.uint8)
            Image.fromarray(new_train_image).save(os.path.join("atsds_large_all_background/imgs_main_extend", split, train_imgs_folder, train_imgs_path[idx]))