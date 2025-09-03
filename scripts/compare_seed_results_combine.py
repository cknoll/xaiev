#!/usr/bin/env python3
"""
Script to automatically compare results.pcl files from different seed runs in XAI evaluation.

This script:
1. Navigates to the XAI_evaluation folder
2. Automatically processes both alexnet_simple and simple_cnn models
3. Finds all possible folder combinations
4. Generates comparison plots for all valid combinations with two curves per plot
5. Saves the combined plots with descriptive naming
"""

import os
import sys
from PIL import Image
from pathlib import Path
from itertools import combinations
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from io import BytesIO


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
                        result_pcl = find_result_pcl(final_path)
                        
                        if result_pcl:
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
    
    # For each path combination, compare data from different model folders
    for sub1, sub2, sub3 in path_combinations:
        # Find all model folders that have this path combination
        valid_model_folders = []
        for model_folder in model_folders:
            final_path = os.path.join(base_folder, model_folder, sub1, sub2, "test", sub3)
            result_pcl = find_result_pcl(final_path)
            
            if result_pcl:
                valid_model_folders.append({
                    'folder': model_folder,
                    'result_pcl': result_pcl
                })
        
        # Generate combinations between different model folders for this path
        for folder1, folder2 in combinations(valid_model_folders, 2):
            # Generate output filename
            output_filename = f"{sub1}_{sub2}_test_{sub3}_vs_{folder1['folder']}_vs_{folder2['folder']}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if output already exists
            if os.path.exists(output_path):
                print(f"Skipping existing: {output_filename}")
                continue
            
            # Create combined plot
            success = create_comparison_plot(
                folder1['result_pcl'], folder2['result_pcl'], 
                folder1['folder'], folder2['folder'],
                sub1, sub2, sub3, output_path
            )
            
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


def find_result_pcl(folder_path):
    """Find results.pcl file in the given folder."""
    result_path = os.path.join(folder_path, "results.pcl")
    if os.path.exists(result_path):
        return result_path
    return None


def extract_data_from_pcl(pcl_path):
    """Extract plottable data from a pickle file following the same pattern as visualize_evaluation."""
    try:
        with open(pcl_path, 'rb') as f:
            pcl_data = pickle.load(f)
        
        # The pickle file should contain a dictionary with XAI method names as keys
        # We need to determine which XAI method to use based on the file path
        xai_method = None
        if 'gradcam' in pcl_path:
            xai_method = 'gradcam'
        elif 'lime' in pcl_path:
            xai_method = 'lime'
        elif 'xrai' in pcl_path:
            xai_method = 'xrai'
        
        if xai_method and xai_method in pcl_data:
            # Extract the tuple data: (correct, correct_5, softmax, score, loss)
            method_tuple = pcl_data[xai_method]
            
            if isinstance(method_tuple, (list, tuple)) and len(method_tuple) >= 1:
                # Get the 'correct' data (first element of the tuple)
                correct = method_tuple[0]
                
                # Process the same way as in visualize_evaluation:
                # accuracy = np.mean((np.divide(correct, 50)), axis=1)
                if isinstance(correct, (list, np.ndarray)):
                    correct = np.array(correct)
                    accuracy = np.mean(np.divide(correct, 50), axis=1) if correct.ndim > 1 else np.divide(correct, 50)
                    
                    # Create x-axis data (percentage range: 0, 10, 20, ..., 100)
                    x_data = list(range(0, 101, 10))
                    y_data = accuracy
                    
                    # Ensure x and y have same length
                    min_len = min(len(x_data), len(y_data))
                    x_data = x_data[:min_len]
                    y_data = y_data[:min_len]
                    
                    return x_data, y_data, xai_method
        
        print(f"Warning: Could not extract data from {pcl_path}")
        return None, None, None
        
    except Exception as e:
        print(f"Error loading {pcl_path}: {e}")
        return None, None, None


def create_comparison_plot(pcl_path1, pcl_path2, folder1, folder2, sub1, sub2, sub3, output_path):
    """Create a comparison plot with two curves from two pickle files."""
    try:
        # Extract data from both pickle files
        x_data1, y_data1, xai_method1 = extract_data_from_pcl(pcl_path1)
        x_data2, y_data2, xai_method2 = extract_data_from_pcl(pcl_path2)
        
        if x_data1 is None or x_data2 is None:
            print(f"Failed to extract data from pickle files")
            return False
        
        # Determine XAI method (should be the same for both)
        xai_method = xai_method1 if xai_method1 else xai_method2
        if not xai_method:
            xai_method = "Unknown"
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot both curves
        plt.plot(x_data1, y_data1, color='blue', label=folder1, linewidth=2, marker='o')
        plt.plot(x_data2, y_data2, color='red', label=folder2, linewidth=2, marker='s')
        
        # Set plot properties
        plt.title(f'{xai_method.upper()} Results Comparison\n{sub1}/{sub2}/test/{sub3}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Percentage', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        return True
        
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        return False


def main():
    """Main function to execute the script."""
    # Step 1: Navigate to the base folder
    base_folder = "/data/horse/ws/luch715g-XAI_workspace/data/atsds_large/XAI_evaluation"
    
    if not os.path.exists(base_folder):
        print(f"Error: Base folder does not exist: {base_folder}")
        sys.exit(1)
    
    print(f"Working in: {base_folder}")
    
    # Create output directory
    output_dir = os.path.join(base_folder, "compare_different_seed_result_plots")
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
