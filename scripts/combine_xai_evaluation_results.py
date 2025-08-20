#!/usr/bin/env python3
"""
Script to combine XAI evaluation result images into a single visualization.

This script:
1. Prompts user to select a folder under /data
2. Prompts user to select a folder from /XAI_evaluation 
3. Prompts user to select method (revelation/occlusion)
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
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]


def select_folder_interactive(base_path, prompt_message):
    """Interactively select a folder from the given base path."""
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


def select_method_interactive():
    """Interactively select evaluation method."""
    methods = ["revelation", "occlusion"]
    
    print("\nSelect evaluation method:")
    for i, method in enumerate(methods, 1):
        print(f"{i}. {method}")
    
    while True:
        try:
            choice = int(input("Enter your choice (number): ")) - 1
            if 0 <= choice < len(methods):
                return methods[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def find_result_images(xai_evaluation_path, method):
    """Find result.png images for each XAI method and baseline type."""
    xai_methods = ["gradcam", "lime", "xrai"]
    baseline_types = ["average", "default", "black"]
    
    images = {}
    
    for xai_method in xai_methods:
        images[xai_method] = {}
        xai_method_path = os.path.join(xai_evaluation_path, xai_method)
        
        if not os.path.exists(xai_method_path):
            print(f"Warning: {xai_method_path} does not exist")
            continue
            
        for baseline_type in baseline_types:
            baseline_path = os.path.join(xai_method_path, baseline_type)
            result_image_path = os.path.join(baseline_path, method, "result.png")
            
            if os.path.exists(result_image_path):
                images[xai_method][baseline_type] = result_image_path
            else:
                print(f"Warning: {result_image_path} does not exist")
                images[xai_method][baseline_type] = None
    
    return images


def create_combined_image(images, output_path, spacing=30):
    """Create a combined image with 3x3 grid layout."""
    xai_methods = ["gradcam", "lime", "xrai"]
    baseline_types = ["average", "default", "black"]
    
    # Load all images and get dimensions
    loaded_images = {}
    max_width = 0
    max_height = 0
    
    for xai_method in xai_methods:
        loaded_images[xai_method] = {}
        for baseline_type in baseline_types:
            if images[xai_method].get(baseline_type):
                try:
                    img = Image.open(images[xai_method][baseline_type])
                    loaded_images[xai_method][baseline_type] = img
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
                except Exception as e:
                    print(f"Error loading {images[xai_method][baseline_type]}: {e}")
                    loaded_images[xai_method][baseline_type] = None
            else:
                loaded_images[xai_method][baseline_type] = None
    
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
        for col, baseline_type in enumerate(baseline_types):
            if loaded_images[xai_method][baseline_type]:
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                combined_image.paste(loaded_images[xai_method][baseline_type], (x, y))
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
    parser = argparse.ArgumentParser(description="Combine XAI evaluation result images")
    parser.add_argument("--data-folder", help="Folder under data directory to use")
    parser.add_argument("--xai-evaluation-folder", help="Folder under /XAI_evaluation to use")
    parser.add_argument("--method", choices=["revelation", "occlusion"], help="Evaluation method")
    parser.add_argument("--output", help="Output file path", default="combined_xai_results.png")
    parser.add_argument("--data-base-path", help="Base path for data folders", default="data")
    
    args = parser.parse_args()
    
    # Step 1: Select folder under data directory
    data_base_path = args.data_base_path
    if args.data_folder:
        data_folder = args.data_folder
        if not os.path.exists(os.path.join(data_base_path, data_folder)):
            print(f"Error: Folder {data_folder} does not exist in {data_base_path}")
            print(f"Available folders in {data_base_path}:")
            available = get_available_folders(data_base_path)
            if available:
                for folder in available:
                    print(f"  - {folder}")
            else:
                print(f"  No folders found in {data_base_path}")
            return 1
    else:
        data_folder = select_folder_interactive(data_base_path, f"Select a folder under {data_base_path}:")
        if not data_folder:
            print("No folder selected. Exiting.")
            return 1
    
    data_folder_path = os.path.join(data_base_path, data_folder)
    print(f"Selected data folder: {data_folder_path}")
    
    # Step 2: Select folder from /XAI_evaluation
    xai_evaluation_base_path = os.path.join(data_folder_path, "XAI_evaluation")
    if args.xai_evaluation_folder:
        xai_evaluation_folder = args.xai_evaluation_folder
        if not os.path.exists(os.path.join(xai_evaluation_base_path, xai_evaluation_folder)):
            print(f"Error: Folder {xai_evaluation_folder} does not exist in {xai_evaluation_base_path}")
            return 1
    else:
        xai_evaluation_folder = select_folder_interactive(
            xai_evaluation_base_path, 
            "Select a folder from /XAI_evaluation:"
        )
        if not xai_evaluation_folder:
            print("No folder selected. Exiting.")
            return 1
    
    xai_evaluation_path = os.path.join(xai_evaluation_base_path, xai_evaluation_folder)
    print(f"Selected XAI evaluation folder: {xai_evaluation_path}")
    
    # Step 3: Select method
    if args.method:
        method = args.method
    else:
        method = select_method_interactive()
    
    print(f"Selected method: {method}")
    
    # Step 4-6: Find and combine images
    print("Finding result images...")
    images = find_result_images(xai_evaluation_path, method)
    
    # Create output filename if not specified
    if args.output == "combined_xai_results.png":
        output_filename = f"combined_{data_folder}_{xai_evaluation_folder}_{method}.png"
    else:
        output_filename = args.output
    
    print("Creating combined image...")
    success = create_combined_image(images, output_filename, spacing=30)
    
    if success:
        print("Task completed successfully!")
        return 0
    else:
        print("Task failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
