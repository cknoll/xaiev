#!/usr/bin/env python3
"""
Script to compare result.png files from different seed runs in XAI evaluation.

This script:
1. Navigates to the XAI_evaluation folder
2. Prompts user to choose between alexnet_simple or simple_cnn
3. Guides user through folder selection process
4. Finds two result.png files and combines them
5. Saves the combined image with descriptive naming
"""

import os
import sys
from PIL import Image
from pathlib import Path


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_user_choice(prompt, options):
    """Get user choice from a list of options."""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    while True:
        try:
            choice = int(input("Enter your choice (number): ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number")


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
    """Find result.png file in the given folder."""
    result_path = os.path.join(folder_path, "result.png")
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
    
    # Step 2: Ask user to choose between alexnet_simple or simple_cnn
    model_options = ["alexnet_simple", "simple_cnn"]
    chosen_model = get_user_choice("Choose a model:", model_options)
    print(f"Selected model: {chosen_model}")
    
    # Step 3: Find folders starting with the chosen model
    model_folders = get_folders_starting_with(base_folder, chosen_model)
    
    if len(model_folders) < 2:
        print(f"Error: Need at least 2 folders starting with '{chosen_model}', found {len(model_folders)}")
        sys.exit(1)
    
    print(f"Found {len(model_folders)} folders starting with '{chosen_model}'")
    
    # We need to navigate through both folders in parallel
    selected_paths = []
    
    for i, folder in enumerate(model_folders[:2]):  # Take first 2 folders
        current_path = os.path.join(base_folder, folder)
        print(f"\n--- Processing folder {i+1}: {folder} ---")
        
        # Step 4: Choose first subfolder level
        subfolders1 = get_subfolders(current_path)
        if not subfolders1:
            print(f"No subfolders found in {current_path}")
            sys.exit(1)
        
        chosen_subfolder1 = get_user_choice(f"Choose subfolder in {folder}:", subfolders1)
        current_path = os.path.join(current_path, chosen_subfolder1)
        
        # Step 5: Choose second subfolder level
        subfolders2 = get_subfolders(current_path)
        if not subfolders2:
            print(f"No subfolders found in {current_path}")
            sys.exit(1)
        
        chosen_subfolder2 = get_user_choice(f"Choose second level subfolder:", subfolders2)
        current_path = os.path.join(current_path, chosen_subfolder2)
        
        # Step 6: Choose third subfolder level
        subfolders3 = get_subfolders(current_path)
        if not subfolders3:
            print(f"No subfolders found in {current_path}")
            sys.exit(1)
        
        chosen_subfolder3 = get_user_choice(f"Choose third level subfolder:", subfolders3)
        current_path = os.path.join(current_path, chosen_subfolder3)
        
        selected_paths.append({
            'path': current_path,
            'folder': folder,
            'sub1': chosen_subfolder1,
            'sub2': chosen_subfolder2,
            'sub3': chosen_subfolder3
        })
    
    # Step 7: Find result.png files in both selected paths
    result_images = []
    for path_info in selected_paths:
        result_path = find_result_png(path_info['path'])
        if result_path:
            result_images.append(result_path)
            print(f"Found result.png in: {path_info['path']}")
        else:
            print(f"Error: result.png not found in: {path_info['path']}")
            sys.exit(1)
    
    if len(result_images) != 2:
        print(f"Error: Expected 2 result.png files, found {len(result_images)}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.join(base_folder, "compare_different_seed_result")
    ensure_dir(output_dir)
    
    # Generate output filename based on user choices
    path1_info = selected_paths[0]
    path2_info = selected_paths[1]
    
    output_filename = f"{chosen_model}_{path1_info['sub1']}_{path1_info['sub2']}_{path1_info['sub3']}_vs_{path2_info['sub1']}_{path2_info['sub2']}_{path2_info['sub3']}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # Combine the images
    success = combine_images(result_images[0], result_images[1], output_path, gap=20)
    
    if success:
        print(f"\nSuccess! Combined image created:")
        print(f"Output: {output_path}")
        print(f"Comparing:")
        print(f"  - {selected_paths[0]['folder']}/{path1_info['sub1']}/{path1_info['sub2']}/{path1_info['sub3']}")
        print(f"  - {selected_paths[1]['folder']}/{path2_info['sub1']}/{path2_info['sub2']}/{path2_info['sub3']}")
    else:
        print("Failed to create combined image")
        sys.exit(1)


if __name__ == "__main__":
    main()
