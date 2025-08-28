#!/usr/bin/env python3
"""
Script to combine images from evaluation folders.

This script:
1. Goes to /data/horse/ws/luch715g-XAI_workspace/data/atsds_large/XAI_evaluation
2. Finds pairs of subfolders whose names start with "alexnet_simple" and "simple_cnn"
3. Creates new output folders with the same structure
4. Searches recursively for matching images and combines them with 20px gap
5. Ignores files named "result.png"
"""

import os
import glob
from PIL import Image
from pathlib import Path


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def find_folder_pairs(base_path, prefix):
    """Find folders starting with the given prefix in the base path."""
    pattern = os.path.join(base_path, f"{prefix}*")
    folders = glob.glob(pattern)
    folders = [f for f in folders if os.path.isdir(f)]
    
    if len(folders) != 2:
        raise ValueError(f"Expected exactly 2 folders starting with '{prefix}', found {len(folders)}: {folders}")
    
    return sorted(folders)


def get_relative_structure(folder_path, base_path):
    """Get all subdirectories relative to base_path."""
    subdirs = []
    for root, dirs, files in os.walk(folder_path):
        rel_path = os.path.relpath(root, base_path)
        if rel_path != '.':
            subdirs.append(rel_path)
    return subdirs


def find_matching_images(folder1, folder2):
    """Find all matching image files between two folder structures."""
    matches = []
    
    # Walk through folder1 and look for matching files in folder2
    for root, dirs, files in os.walk(folder1):
        for file in files:
            # Skip result.png files
            if file == "result.png":
                continue
                
            # Check if it's an image file
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Get relative path from folder1 base
                rel_path = os.path.relpath(root, folder1)
                
                # Construct corresponding path in folder2
                corresponding_dir = os.path.join(folder2, rel_path)
                corresponding_file = os.path.join(corresponding_dir, file)
                
                # Check if matching file exists in folder2
                if os.path.exists(corresponding_file):
                    img1_path = os.path.join(root, file)
                    matches.append((img1_path, corresponding_file, rel_path, file))
    
    return matches


def combine_images(img_path1, img_path2, output_path, gap=20):
    """Combine two images horizontally with a gap between them."""
    try:
        # Open both images
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)
        
        # Get dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size
        
        # Calculate combined dimensions
        max_height = max(height1, height2)
        total_width = width1 + width2 + gap
        
        # Create new image with white background
        combined = Image.new('RGB', (total_width, max_height), 'white')
        
        # Paste images
        combined.paste(img1, (0, 0))
        combined.paste(img2, (width1 + gap, 0))
        
        # Save combined image
        ensure_dir(os.path.dirname(output_path))
        combined.save(output_path)
        
        print(f"Combined: {os.path.basename(img_path1)} -> {output_path}")
        
    except Exception as e:
        print(f"Error combining {img_path1} and {img_path2}: {e}")


def create_output_structure(output_base, folder1, folder2):
    """Create output directory structure based on input folders."""
    # Get all subdirectories from both folders
    subdirs1 = get_relative_structure(folder1, folder1)
    subdirs2 = get_relative_structure(folder2, folder2)
    
    # Combine and deduplicate
    all_subdirs = list(set(subdirs1 + subdirs2))
    
    # Create all subdirectories in output
    for subdir in all_subdirs:
        output_dir = os.path.join(output_base, subdir)
        ensure_dir(output_dir)


def process_folder_pair(base_path, prefix):
    """Process a pair of folders with the given prefix."""
    print(f"\n=== Processing {prefix} folders ===")
    
    try:
        # Find the two folders with the given prefix
        folders = find_folder_pairs(base_path, prefix)
        folder1, folder2 = folders
        
        print(f"Found folders:")
        print(f"  Folder 1: {folder1}")
        print(f"  Folder 2: {folder2}")
        
        # Create output folder name
        folder1_name = os.path.basename(folder1)
        folder2_name = os.path.basename(folder2)
        output_folder_name = f"combined_{folder1_name}_{folder2_name}"
        output_path = os.path.join(base_path, output_folder_name)
        
        print(f"Output folder: {output_path}")
        
        # Create output directory structure
        create_output_structure(output_path, folder1, folder2)
        
        # Find matching images
        matches = find_matching_images(folder1, folder2)
        print(f"Found {len(matches)} matching image pairs")
        
        # Process each matching pair
        processed_count = 0
        for img1_path, img2_path, rel_path, filename in matches:
            output_img_dir = os.path.join(output_path, rel_path)
            output_img_path = os.path.join(output_img_dir, filename)
            
            combine_images(img1_path, img2_path, output_img_path, gap=20)
            processed_count += 1
        
        print(f"Processing complete for {prefix}! Combined {processed_count} images saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing {prefix} folders: {e}")


def main():
    # Base path
    base_path = "/data/horse/ws/luch715g-XAI_workspace/data/atsds_large/XAI_evaluation"
    
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    print(f"Processing folders in: {base_path}")
    
    # Process alexnet_simple folders
    process_folder_pair(base_path, "alexnet_simple")
    
    # Process simple_cnn folders
    process_folder_pair(base_path, "simple_cnn")
    
    print("\n=== All processing complete! ===")


if __name__ == "__main__":
    main()
