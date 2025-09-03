#!/usr/bin/env python3
"""
Script to automatically compare results.png files from different seed runs in XAI evaluation.

This script:
1. Navigates to the XAI_evaluation folder
2. Automatically processes both alexnet_simple and simple_cnn models
3. Finds all possible folder combinations
4. Generates comparison images for all valid combinations
5. Saves the combined images with descriptive naming
"""

import os
import sys
from PIL import Image
from pathlib import Path
from itertools import combinations


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_all_folder_path_combinations(base_folder, model_prefix):
    """Get all unique folder path combinations (sub1, sub2, sub3) for a given model prefix."""
    path_combinations = set()
    model_folders = get_folders_starting_with(base_folder, model_prefix)
    
    for model_folder in model_folders:
        model_path = os.path.join(base_folder, model_folder)
        
        # Get first level subfolders
        subfolders1 = get_subfolders(model_path)
        for sub1 in subfolders1:
            sub1_path = os.path.join(model_path, sub1)
            
            # Get second level subfolders
            subfolders2 = get_subfolders(sub1_path)
            for sub2 in subfolders2:
                test_path = os.path.join(sub1_path, sub2, "test")
                
                if os.path.exists(test_path):
                    # Get third level subfolders
                    subfolders3 = get_subfolders(test_path)
                    for sub3 in subfolders3:
                        final_path = os.path.join(test_path, sub3)
                        result_png = find_result_png(final_path)
                        
                        if result_png:
                            path_combinations.add((sub1, sub2, sub3))
    
    return sorted(list(path_combinations))


def process_model(base_folder, model_prefix, output_dir):
    """Process all combinations for a specific model."""
    print(f"\nProcessing model: {model_prefix}")
    
    # Get all model folders for this prefix
    model_folders = get_folders_starting_with(base_folder, model_prefix)
    
    if len(model_folders) < 2:
        print(f"Skipping {model_prefix}: Need at least 2 model folders, found {len(model_folders)}")
        return 0
    
    print(f"Found {len(model_folders)} model folders for {model_prefix}")
    
    # Get all unique path combinations
    path_combinations = get_all_folder_path_combinations(base_folder, model_prefix)
    
    if not path_combinations:
        print(f"No valid path combinations found for {model_prefix}")
        return 0
    
    print(f"Found {len(path_combinations)} unique path combinations")
    
    combinations_count = 0
    
    # For each path combination, compare images from different model folders
    for sub1, sub2, sub3 in path_combinations:
        # Find all model folders that have this path combination
        valid_model_folders = []
        for model_folder in model_folders:
            final_path = os.path.join(base_folder, model_folder, sub1, sub2, "test", sub3)
            result_png = find_result_png(final_path)
            
            if result_png:
                valid_model_folders.append({
                    'folder': model_folder,
                    'result_png': result_png
                })
        
        # Generate combinations between different model folders for this path
        for folder1, folder2 in combinations(valid_model_folders, 2):
            # Generate output filename
            output_filename = f"{model_prefix}_{sub1}_{sub2}_test_{sub3}_vs_{folder1['folder']}_vs_{folder2['folder']}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping existing: {output_filename}")
                continue
            
            # Combine the images
            success = combine_images(folder1['result_png'], folder2['result_png'], output_path, gap=20)
            
            if success:
                combinations_count += 1
                print(f"Created: {output_filename}")
            else:
                print(f"Failed to create: {output_filename}")
    
    return combinations_count


def get_folders_starting_with(base_path, prefix):
    """Get all folders in base_path that start with the given prefix."""
    if not os.path.exists(base_path):
        return []
    
    folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith(prefix):
            folders.append(item)
    
    return sorted(folders)


def get_subfolders(path):
    """Get all subfolders in the given path."""
    if not os.path.exists(path):
        return []
    
    folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    return sorted(folders)


def find_result_png(folder_path):
    """Find results.png file in the given folder."""
    result_path = os.path.join(folder_path, "results.png")
    if os.path.exists(result_path):
        return result_path
    return None


def combine_images(img_path1, img_path2, output_path, gap=20):
    """Combine two images horizontally with a gap."""
    try:
        # Open both images
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        
        # Get dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Calculate combined image dimensions
        max_height = max(height1, height2)
        total_width = width1 + width2 + gap
        
        # Create new image with white background
        combined = Image.new('RGB', (total_width, max_height), 'white')
        
        # Paste images
        combined.paste(img1, (0, 0))
        combined.paste(img2, (width1 + gap, 0))
        
        # Save combined image
        combined.save(output_path)
        print(f"Combined image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error combining images: {e}")
        return False
    
    return True


def main():
    """Main function to execute the script."""
    # Step 1: Navigate to the base folder
    base_folder = "/data/horse/ws/luch715g-XAI_workspace/data/atsds_large/XAI_evaluation"
    
    if not os.path.exists(base_folder):
        print(f"Error: Base folder does not exist: {base_folder}")
        sys.exit(1)
    
    print(f"Working in: {base_folder}")
    
    # Create output directory
    output_dir = os.path.join(base_folder, "compare_different_seed_result")
    ensure_dir(output_dir)
    print(f"Output directory: {output_dir}")
    
    # Process both model types automatically
    model_options = ["alexnet_simple", "simple_cnn"]
    total_combinations = 0
    
    for model_prefix in model_options:
        combinations_count = process_model(base_folder, model_prefix, output_dir)
        total_combinations += combinations_count
    
    print(f"\n=== Summary ===")
    print(f"Total combinations created: {total_combinations}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
