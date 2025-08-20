#!/usr/bin/env python3
"""
Script to combine XAI evaluation result images into a single visualization.

This script:
1. Prompts user to select a folder under /data
2. Prompts user to select a folder from /XAI_evaluation 
3. Prompts user to select evaluation method (revelation/occlusion)
4. Extracts result.png images from gradcam, lime, and xrai subfolders
5. Combines them into a 3x3 grid with proper spacing
"""

import os
import sys
from PIL import Image
import argparse


def get_available_folders(base_path):
    """Get list of available folders in the given path."""
    if not os.path.exists(base_path):
        return []
    
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    return sorted(folders)


def select_folder_interactive(base_path, prompt_message):
    """Interactive folder selection."""
    folders = get_available_folders(base_path)
    
    if not folders:
        print(f"No folders found in {base_path}")
        return None
    
    print(f"\n{prompt_message}")
    for i, folder in enumerate(folders, 1):
        print(f"{i}. {folder}")
    
    while True:
        try:
            choice = input(f"Select folder (1-{len(folders)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                return folders[idx]
            else:
                print(f"Please enter a number between 1 and {len(folders)}")
        except ValueError:
            print("Please enter a valid number")


def select_method_interactive():
    """Interactive method selection."""
    methods = ["revelation", "occlusion"]
    
    print("\nSelect evaluation method:")
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method}")
    
    while True:
        try:
            choice = input(f"Select method (1-{len(methods)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(methods):
                return methods[idx]
            else:
                print(f"Please enter a number between 1 and {len(methods)}")
        except ValueError:
            print("Please enter a valid number")


def find_result_images(xai_evaluation_path, method):
    """Find result.png images for each XAI method and condition."""
    xai_methods = ["gradcam", "lime", "xrai"]
    conditions = ["average", "default", "black"]
    
    images = {}
    
    for xai_method in xai_methods:
        images[xai_method] = {}
        
        for condition in conditions:
            # Construct the path to the result.png file
            result_path = os.path.join(
                xai_evaluation_path, 
                xai_method, 
                method, 
                condition, 
                "result.png"
            )
            
            if os.path.exists(result_path):
                images[xai_method][condition] = result_path
                print(f"Found: {result_path}")
            else:
                print(f"Warning: Missing {result_path}")
                images[xai_method][condition] = None
    
    return images


def create_combined_image(images, output_path, spacing=30):
    """Create a combined image with 3x3 grid layout."""
    xai_methods = ["gradcam", "lime", "xrai"]
    conditions = ["average", "default", "black"]
    
    # Load all images and get dimensions
    loaded_images = {}
    max_width = 0
    max_height = 0
    
    for xai_method in xai_methods:
        loaded_images[xai_method] = {}
        for condition in conditions:
            if images[xai_method][condition]:
                try:
                    img = Image.open(images[xai_method][condition])
                    loaded_images[xai_method][condition] = img
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
                except Exception as e:
                    print(f"Error loading {images[xai_method][condition]}: {e}")
                    loaded_images[xai_method][condition] = None
            else:
                loaded_images[xai_method][condition] = None
    
    if max_width == 0 or max_height == 0:
        print("Error: No valid images found")
        return False
    
    # Calculate combined image dimensions
    combined_width = 3 * max_width + 2 * spacing
    combined_height = 3 * max_height + 2 * spacing
    
    # Create white background
    combined_image = Image.new('RGB', (combined_width, combined_height), 'white')
    
    # Place images in grid
    for row, xai_method in enumerate(xai_methods):
        for col, condition in enumerate(conditions):
            if loaded_images[xai_method][condition]:
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                
                # Resize image to max dimensions if needed
                img = loaded_images[xai_method][condition]
                if img.size != (max_width, max_height):
                    img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                
                combined_image.paste(img, (x, y))
            else:
                # Create placeholder for missing image
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                placeholder = Image.new('RGB', (max_width, max_height), 'lightgray')
                combined_image.paste(placeholder, (x, y))
    
    # Save combined image
    try:
        combined_image.save(output_path)
        print(f"Combined image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving combined image: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Combine XAI evaluation result images')
    parser.add_argument('--data-path', default='/data', 
                       help='Base path to data directory (default: /data)')
    parser.add_argument('--output', default='combined_xai_results.png',
                       help='Output filename (default: combined_xai_results.png)')
    
    args = parser.parse_args()
    
    # Step 1: Select folder under /data
    data_path = args.data_path
    if not os.path.exists(data_path):
        print(f"Error: Data path {data_path} does not exist")
        sys.exit(1)
    
    selected_data_folder = select_folder_interactive(
        data_path, 
        f"Available folders under {data_path}:"
    )
    
    if not selected_data_folder:
        print("No folder selected. Exiting.")
        sys.exit(1)
    
    # Step 2: Select folder from /XAI_evaluation
    data_folder_path = os.path.join(data_path, selected_data_folder)
    xai_evaluation_base = os.path.join(data_folder_path, "XAI_evaluation")
    
    if not os.path.exists(xai_evaluation_base):
        print(f"Error: XAI_evaluation folder not found at {xai_evaluation_base}")
        sys.exit(1)
    
    selected_xai_folder = select_folder_interactive(
        xai_evaluation_base,
        f"Available folders under {xai_evaluation_base}:"
    )
    
    if not selected_xai_folder:
        print("No XAI evaluation folder selected. Exiting.")
        sys.exit(1)
    
    # Step 3: Select method (revelation/occlusion)
    method = select_method_interactive()
    
    # Construct full path to selected XAI evaluation folder
    xai_evaluation_path = os.path.join(xai_evaluation_base, selected_xai_folder)
    
    print(f"\nProcessing XAI evaluation results from: {xai_evaluation_path}")
    print(f"Using method: {method}")
    
    # Step 4-6: Find and combine images
    images = find_result_images(xai_evaluation_path, method)
    
    # Create output filename with context
    output_filename = f"combined_{selected_data_folder}_{selected_xai_folder}_{method}.png"
    
    if create_combined_image(images, output_filename, spacing=30):
        print(f"\nSuccess! Combined image created: {output_filename}")
    else:
        print("\nFailed to create combined image")
        sys.exit(1)


if __name__ == "__main__":
    main()
