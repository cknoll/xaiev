import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont

# ============================
#        CONFIG
# ============================
dataset_name = "geometry_512"
text_size = 50 # text size for watermark
alpha_value = 180 # transparency for watermark (0-255)
position = (100, 220)  # Fixed position for watermark
input_root = f"{dataset_name}/imgs_main"                   # Path to original dataset (no watermark)
output_root = f"{dataset_name}_{text_size}_{alpha_value}/imgs_main"          # Where copied + watermarked data will be saved
watermark_text = 'parallelogram' # Text to use as watermark
text_color = (255, 255, 255, alpha_value)                        # Color (white) for watermark text
apply_fraction = 1                                       # 100% of images will get watermark
splits = ['train', 'test']                               # Process both train and test
target_subfolder = "06_parallelogram"                    # Only process this subfolder (same as before)

# ============================
#        LOAD FONT
# ============================
font = None
font_paths = [
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/arial.ttf",
    "/System/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

for fpath in font_paths:
    try:
        font = ImageFont.truetype(fpath, text_size)
        print(f"✔️ Using font: {fpath}")
        break
    except:
        pass

if font is None:
    font = ImageFont.load_default()
    print("⚠️ Using default font")


# ============================
#     FUNCTION: WATERMARK
# ============================
def apply_watermark(img_path, out_path):
    img = Image.open(img_path).convert("RGBA")

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fixed position (same as your original script)
    draw.text(position, watermark_text, font=font, fill=text_color)

    # Merge and save
    watermarked = Image.alpha_composite(img, overlay)
    watermarked.convert("RGB").save(out_path)


# ============================
#     MAIN: COPY + WATERMARK
# ============================
print("\n===============================")
print("   START COPYING + WATERMARK   ")
print("===============================\n")

for split in splits:
    src_folder = os.path.join(input_root, split)
    dst_folder = os.path.join(output_root, split)

    print(f"📁 Copying: {src_folder} → {dst_folder}")

    # Allows overwriting existing directory (Python 3.8+)
    shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
    print(f"✔️ Done copying {split}\n")

    # Locate the target subfolder for watermarking
    sub_src = os.path.join(output_root, split, target_subfolder)

    if not os.path.exists(sub_src):
        print(f"❌ WARNING: folder not found: {sub_src}")
        continue

    images = [f for f in os.listdir(sub_src)
              if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"🖼️ Found {len(images)} images in {split}/{target_subfolder}")

    num_to_watermark = int(len(images) * apply_fraction)
    watermark_list = set(random.sample(images, num_to_watermark))

    for img_name in images:
        in_path = os.path.join(sub_src, img_name)
        out_path = os.path.join(sub_src, img_name)

        apply_watermark(in_path, out_path)

    print(f"✔️ Added watermark to {num_to_watermark} images in {split}\n")

print("\n🎉 ALL DONE! Train + Test have been copied and watermarked")
