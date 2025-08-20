#!/usr/bin/env python3
"""
Script to combine XAI evaluation results into a 3x3 grid.

This script:
1. Finds folders under data/ and asks user to choose via command line args
2. Navigates to XAI_evaluation folder and asks user to choose interactively
3. Extracts result.png files from gradcam/, lime/, and xrai/ folders
4. Creates a 3x3 grid with proper spacing and ordering
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys


def get_available_folders(base_path):
    """Get list of available folders in the given path."""
    if not os.path.exists(base_path):
        return []
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]


def select_folder_interactive(base_path, prompt_message):
    """Interactively select a folder from available options."""
    folders = get_available_folders(base_path)
    if not folders:
        print(f"No folders found in {base_path}")
        return None
    
    print(f"\n{prompt_message}")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = int(input("Enter your choice (number): ")) - 1
            if 0 <= choice < len(folders):
                return folders[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def extract_result_images(base_path, method):
    """Extract result.png files from the XAI evaluation structure."""
    images = {}
    xai_methods = ['gradcam', 'lime', 'xrai']
    subfolder_order = ['average', 'default', 'black']  # Order for grid placement
    
    for xai_method in xai_methods:
        images[xai_method] = {}
        xai_path = os.path.join(base_path, xai_method)
        
        if not os.path.exists(xai_path):
            print(f"Warning: {xai_method} folder not found at {xai_path}")
            continue
            
        # Get all subfolders under the XAI method folder
        subfolders = get_available_folders(xai_path)
        
        for subfolder in subfolders:
            result_path = os.path.join(xai_path, subfolder, 'test', method, 'results.png')
            
            if os.path.exists(result_path):
                try:
                    img = Image.open(result_path)
                    images[xai_method][subfolder] = img.copy()
                    print(f"Loaded: {result_path}")
                except Exception as e:
                    print(f"Error loading {result_path}: {e}")
            else:
                print(f"Warning: result.png not found at {result_path}")
    
    return images


def create_3x3_grid(images, output_path, spacing=30):
    """Create a 3x3 grid of images with proper spacing."""
    xai_methods = ['gradcam', 'lime', 'xrai']
    subfolder_order = ['average', 'default', 'black']
    
    # Find the maximum image dimensions
    max_width = 0
    max_height = 0
    
    for xai_method in xai_methods:
        for subfolder in subfolder_order:
            if xai_method in images and subfolder in images[xai_method]:
                img = images[xai_method][subfolder]
                max_width = max(max_width, img.width)
                max_height = max(max_height, img.height)
    
    if max_width == 0 or max_height == 0:
        print("Error: No valid images found to create grid")
        return False
    
    # Calculate header height for column labels
    header_height = 50
    
    # Calculate grid dimensions
    grid_width = 3 * max_width + 4 * spacing  # 3 images + 4 spacing areas (left, 2 middle, right)
    grid_height = 3 * max_height + 4 * spacing + header_height  # 3 rows + 4 spacing areas + header
    
    # Create white background
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_image)
    
    # Try to use a default font, fallback to basic font if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            # Try other common font paths
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
    
    # Add column headers
    for col, subfolder in enumerate(subfolder_order):
        x = spacing + col * (max_width + spacing) + max_width // 2
        y = header_height // 2
        
        if font:
            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), subfolder, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x -= text_width // 2
            y -= text_height // 2
            draw.text((x, y), subfolder, fill='black', font=font)
        else:
            # Fallback without font
            draw.text((x - len(subfolder) * 3, y), subfolder, fill='black')
    
    # Place images in grid
    for row, xai_method in enumerate(xai_methods):
        for col, subfolder in enumerate(subfolder_order):
            if xai_method in images and subfolder in images[xai_method]:
                img = images[xai_method][subfolder]
                
                # Calculate position (offset by header height)
                x = spacing + col * (max_width + spacing)
                y = spacing + header_height + row * (max_height + spacing)
                
                # Center the image if it's smaller than max dimensions
                if img.width < max_width:
                    x += (max_width - img.width) // 2
                if img.height < max_height:
                    y += (max_height - img.height) // 2
                
                grid_image.paste(img, (x, y))
                print(f"Placed {xai_method}/{subfolder} at position ({x}, {y})")
            else:
                print(f"Warning: Missing image for {xai_method}/{subfolder}")
    
    # Save the grid
    try:
        grid_image.save(output_path)
        print(f"Grid saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving grid: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Combine XAI evaluation results into a 3x3 grid')
    parser.add_argument('--data-folder', help='Folder name under data/ to process')
    parser.add_argument('--method', choices=['occlusion', 'revelation'], 
                       help='Evaluation method to use (occlusion or revelation)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output filename for the combined grid (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Step 1: Check if data folder exists
    data_base_path = '/data/horse/ws/luch715g-XAI_workspace/data'
    if not os.path.exists(data_base_path):
        print(f"Error: data/ folder not found in current directory")
        return 1
    
    # Check if specified folder exists under data/
    selected_data_folder = os.path.join(data_base_path, args.data_folder)
    if not os.path.exists(selected_data_folder):
        available_folders = get_available_folders(data_base_path)
        print(f"Error: Folder '{args.data_folder}' not found under data/")
        print(f"Available folders: {', '.join(available_folders)}")
        return 1
    
    print(f"Using data folder: {selected_data_folder}")
    
    # Step 2: Navigate to XAI_evaluation
    xai_eval_path = os.path.join(selected_data_folder, 'XAI_evaluation')
    if not os.path.exists(xai_eval_path):
        print(f"Error: XAI_evaluation folder not found at {xai_eval_path}")
        return 1
    
    # Step 3: Interactive folder selection under XAI_evaluation
    selected_eval_folder = select_folder_interactive(
        xai_eval_path, 
        "Choose a folder from XAI_evaluation:"
    )
    
    if not selected_eval_folder:
        print("No folder selected. Exiting.")
        return 1
    
    eval_folder_path = os.path.join(xai_eval_path, selected_eval_folder)
    print(f"Selected evaluation folder: {eval_folder_path}")
    
    # Generate output filename if not provided
    if args.output is None:
        args.output = f"{args.data_folder}_{selected_eval_folder}_{args.method}_compare.png"
    
    # Steps 4-7: Extract images from gradcam, lime, and xrai folders
    images = extract_result_images(eval_folder_path, args.method)
    
    # Verify we have some images
    total_images = sum(len(method_images) for method_images in images.values())
    if total_images == 0:
        print("Error: No result.png files found in the expected structure")
        return 1
    
    print(f"Found {total_images} images total")
    
    # Step 8: Create 3x3 grid
    success = create_3x3_grid(images, args.output)
    
    if success:
        print(f"Successfully created grid: {args.output}")
        return 0
    else:
        print("Failed to create grid")
        return 1


if __name__ == '__main__':
    sys.exit(main())
