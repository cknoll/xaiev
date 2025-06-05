import os
from PIL import Image


folder_A = './geometry_512/imgs_main/test'
folder_B = './geometry_512/XAI_results/alexnet_simple/lime/test'
output_folder = './geometry_512/compare'
gap = 20

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def combine_images(img_path_A, img_path_B):
    img_A = Image.open(img_path_A)
    img_B = Image.open(img_path_B)

    width_A, height_A = img_A.size
    width_B, height_B = img_B.size
    
    new_width = width_A + width_B + gap
    new_height = max(height_A, height_B)
    
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    y_offset_A = (new_height - height_A) // 2
    y_offset_B = (new_height - height_B) // 2

    new_img.paste(img_A, (0, y_offset_A))
    new_img.paste(img_B, (width_A + gap, y_offset_B))

    return new_img

def process_folders(folder_A, folder_B, output_folder):
    ensure_dir(output_folder)

    subfolders = os.listdir(folder_A)
    for sub in subfolders:
        sub_A = os.path.join(folder_A, sub)
        sub_B = os.path.join(folder_B, sub)
        out_sub = os.path.join(output_folder, sub)
        if os.path.isdir(sub_A) and os.path.isdir(sub_B):
            ensure_dir(out_sub)
            filenames = os.listdir(sub_A)
            for fname in filenames:
                img_A_path = os.path.join(sub_A, fname)
                img_B_path_temp = os.path.join(sub_B, 'mask_on_image')
                img_B_path = os.path.join(img_B_path_temp, fname)
                # print(img_A_path)
                # print(img_B_path)
                if os.path.isfile(img_A_path) and os.path.isfile(img_B_path):
                    try:
                        combined = combine_images(img_A_path, img_B_path)
                        save_path = os.path.join(out_sub, fname)
                        combined.save(save_path)
                        print(f"Saved: {save_path}")
                    except Exception as e:
                        print(f"Error processing {fname}: {e}")


process_folders(folder_A, folder_B, output_folder)
