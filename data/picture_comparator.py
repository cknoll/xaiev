import os
from PIL import Image
from torchvision import transforms as transforms

folder_A = './atsds_large/imgs_main/test'
folder_B = './atsds_large/XAI_results/alexnet_simple/gradcam/test'
folder_C = './atsds_large/XAI_results/alexnet_simple/lime/test'
folder_D = './atsds_large/XAI_results/alexnet_simple/xrai/test'
output_folder = './atsds_large/compare'
gap = 20

transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224)),
        ]
    )

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def combine_images(img_path_A, img_path_B, img_path_C, img_path_D):
    img_A = Image.open(img_path_A)
    img_A = transform(img_A)
    img_B = Image.open(img_path_B)
    img_B = transform(img_B)
    img_C = Image.open(img_path_C)
    img_D = Image.open(img_path_D)
    img_D = transform(img_D)

    width_A, height_A = img_A.size
    width_B, height_B = img_B.size
    width_C, height_C = img_C.size
    width_D, height_D = img_D.size
    
    new_width = width_A + width_B + gap
    new_height = height_A + height_B + gap
    
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    y_offset_A = (new_height - height_A) // 2
    y_offset_B = height_A + gap
    x_offset_C = width_A + gap
    x_offset_D = x_offset_C
    y_offset_D = y_offset_B

    new_img.paste(img_A, (0, 0))
    new_img.paste(img_B, (0, y_offset_B))
    new_img.paste(img_C, (x_offset_C, 0))
    new_img.paste(img_D, (x_offset_D, y_offset_D))

    return new_img

def process_folders(folder_A, folder_B, folder_C, folder_D, output_folder):
    ensure_dir(output_folder)

    subfolders = os.listdir(folder_A)
    for sub in subfolders:
        sub_A = os.path.join(folder_A, sub)
        sub_B = os.path.join(folder_B, sub)
        sub_C = os.path.join(folder_C, sub)
        sub_D = os.path.join(folder_D, sub)
        out_sub = os.path.join(output_folder, sub)
        if os.path.isdir(sub_A) and os.path.isdir(sub_B):
            ensure_dir(out_sub)
            filenames = os.listdir(sub_A)
            for fname in filenames:
                img_A_path = os.path.join(sub_A, fname)
                img_B_path_temp = os.path.join(sub_B, 'mask_on_image')
                img_C_path_temp = os.path.join(sub_C, 'mask_on_image')
                img_D_path_temp = os.path.join(sub_D, 'mask_on_image')
                img_B_path = os.path.join(img_B_path_temp, fname)
                img_C_path = os.path.join(img_C_path_temp, fname)
                img_D_path = os.path.join(img_D_path_temp, fname)
                # print(img_A_path)
                # print(img_B_path)
                if os.path.isfile(img_A_path) and os.path.isfile(img_B_path):
                    try:
                        combined = combine_images(img_A_path, img_B_path, img_C_path, img_D_path)
                        save_path = os.path.join(out_sub, fname)
                        combined.save(save_path)
                        print(f"Saved: {save_path}")
                    except Exception as e:
                        print(f"Error processing {fname}: {e}")


process_folders(folder_A, folder_B, folder_C, folder_D, output_folder)
