#!/usr/bin/env python3
"""
Script to combine XAI evaluation results into a single image.

This script:
1. Takes a folder under /data as command line input
2. Selects a folder from /XAI_evaluation 
3. Selects method (revelation/occlusion)
4. Extracts result.png from gradcam/lime/xrai subfolders
5. Combines them into a 3x3 grid with proper spacing
"""

import os
import sys
import argparse
from PIL import Image
import glob

def get_available_data_folders():
    """Get list of available folders under /data"""
    data_path = "/data"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist")
        return []
    
    folders = [f for f in os.listdir(data_path) 
               if os.path.isdir(os.path.join(data_path, f))]
    return sorted(folders)

def get_available_xai_folders(data_folder):
    """Get list of available folders under /data/{data_folder}/XAI_evaluation"""
    xai_path = f"/data/{data_folder}/XAI_evaluation"
    if not os.path.exists(xai_path):
        print(f"Error: {xai_path} does not exist")
        return []
    
    folders = [f for f in os.listdir(xai_path) 
               if os.path.isdir(os.path.join(xai_path, f))]
    return sorted(folders)

def select_folder_interactive(folders, prompt):
    """Interactive folder selection"""
    if not folders:
        print("No folders available")
        return None
    
    print(f"\n{prompt}")
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

def find_result_images(base_path, method):
    """Find result.png images for each XAI method and subfolder type"""
    xai_methods = ['gradcam', 'lime', 'xrai']
    subfolder_types = ['average', 'default', 'black']
    
    images = {}
    
    for xai_method in xai_methods:
        images[xai_method] = {}
        
        for subfolder_type in subfolder_types:
            # Look for result.png in the method-specific subfolder
            pattern = os.path.join(base_path, xai_method, method, subfolder_type, "result.png")
            matches = glob.glob(pattern)
            
            if matches:
                images[xai_method][subfolder_type] = matches[0]
                print(f"Found: {matches[0]}")
            else:
                print(f"Warning: No result.png found at {pattern}")
                images[xai_method][subfolder_type] = None
    
    return images

def create_combined_image(images, output_path, spacing=30):
    """Create a 3x3 grid of images with specified spacing"""
    # Load all images and get dimensions
    loaded_images = {}
    max_width = 0
    max_height = 0
    
    xai_methods = ['gradcam', 'lime', 'xrai']
    subfolder_types = ['average', 'default', 'black']
    
    for xai_method in xai_methods:
        loaded_images[xai_method] = {}
        for subfolder_type in subfolder_types:
            if images[xai_method][subfolder_type]:
                try:
                    img = Image.open(images[xai_method][subfolder_type])
                    loaded_images[xai_method][subfolder_type] = img
                    max_width = max(max_width, img.width)
                    max_height = max(max_height, img.height)
                except Exception as e:
                    print(f"Error loading {images[xai_method][subfolder_type]}: {e}")
                    loaded_images[xai_method][subfolder_type] = None
            else:
                loaded_images[xai_method][subfolder_type] = None
    
    if max_width == 0 or max_height == 0:
        print("Error: No valid images found")
        return False
    
    # Calculate total dimensions
    total_width = 3 * max_width + 2 * spacing
    total_height = 3 * max_height + 2 * spacing
    
    # Create white background
    combined = Image.new('RGB', (total_width, total_height), 'white')
    
    # Place images in grid
    for row, xai_method in enumerate(xai_methods):
        for col, subfolder_type in enumerate(subfolder_types):
            img = loaded_images[xai_method][subfolder_type]
            if img:
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                
                # Resize image to max dimensions if needed
                if img.width != max_width or img.height != max_height:
                    img = img.resize((max_width, max_height), Image.Resampling.LANCZOS)
                
                combined.paste(img, (x, y))
            else:
                # Create a placeholder for missing images
                x = col * (max_width + spacing)
                y = row * (max_height + spacing)
                placeholder = Image.new('RGB', (max_width, max_height), 'lightgray')
                combined.paste(placeholder, (x, y))
    
    # Save the combined image
    try:
        combined.save(output_path)
        print(f"Combined image saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving combined image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Combine XAI evaluation results')
    parser.add_argument('data_folder', help='Folder under /data to process')
    parser.add_argument('method', choices=['revelation', 'occlusion'], 
                       help='Evaluation method to use')
    parser.add_argument('--output', '-o', default='combined_xai_results.png',
                       help='Output filename (default: combined_xai_results.png)')
    parser.add_argument('--spacing', type=int, default=30,
                       help='Spacing between images in pixels (default: 30)')
    
    args = parser.parse_args()
    
    # Verify data folder exists
    data_path = f"/data/{args.data_folder}"
    if not os.path.exists(data_path):
        print(f"Error: Data folder {data_path} does not exist")
        sys.exit(1)
    
    # Get available XAI evaluation folders
    xai_folders = get_available_xai_folders(args.data_folder)
    if not xai_folders:
        print("No XAI evaluation folders found")
        sys.exit(1)
    
    # Interactive selection of XAI folder
    selected_xai_folder = select_folder_interactive(
        xai_folders, 
        "Select XAI evaluation folder:"
    )
    
    if not selected_xai_folder:
        print("No folder selected")
        sys.exit(1)
    
    # Build the base path for finding images
    base_path = f"/data/{args.data_folder}/XAI_evaluation/{selected_xai_folder}"
    
    print(f"\nSearching for result.png files in: {base_path}")
    print(f"Using method: {args.method}")
    
    # Find all result images
    images = find_result_images(base_path, args.method)
    
    # Create combined image
    success = create_combined_image(images, args.output, args.spacing)
    
    if success:
        print(f"\nSuccess! Combined image created: {args.output}")
    else:
        print("\nFailed to create combined image")
        sys.exit(1)

if __name__ == "__main__":
    main()
