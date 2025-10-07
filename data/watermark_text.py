import os
import random
from PIL import Image

# --- Config ---
input_folder = 'atsds_large/imgs_main/train/00031'  # Update this with your actual folder path
output_folder = 'atsds_large/imgs_main/train_w/00031'  # Update as needed
watermark_path = 'dear.png'
resize_size = (80, 40)
apply_fraction = 1  # 80% of images will be watermarked

# --- Ensure output folder exists ---
os.makedirs(output_folder, exist_ok=True)

# --- Load and resize watermark ---
watermark = Image.open(watermark_path).convert("RGBA")
watermark = watermark.resize(resize_size)

alpha = 130 # Set transparency level (0-255)
watermark.putalpha(alpha)

# --- Get all image files ---
all_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
num_to_watermark = int(len(all_images) * apply_fraction)
images_to_watermark = set(random.sample(all_images, num_to_watermark))

# --- Process each image ---
for img_name in all_images:
    img_path = os.path.join(input_folder, img_name)
    img = Image.open(img_path).convert("RGBA")

    if img_name in images_to_watermark:
        # Create transparent overlay and paste watermark at top-left
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay.paste(watermark, (0, 0), watermark)
        img = Image.alpha_composite(img, overlay)

        # Save with "_w" suffix
        base, ext = os.path.splitext(img_name)
        output_name = f"{base}_w{ext}"
    else:
        # Save original image as-is
        output_name = img_name

    # Save as RGB
    output_path = os.path.join(output_folder, output_name)
    img.convert("RGB").save(output_path)

print(f"✔️ Saved all {len(all_images)} images to '{output_folder}'.")
print(f"🖼️ {num_to_watermark} images watermarked with '_w' suffix.")