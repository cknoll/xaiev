#!/usr/bin/env python3
"""
Script to combine XAI evaluation results into a 3x1 grid of plots.

This script:
1. Finds folders under data/ and asks user to choose via command line args
2. Navigates to XAI_evaluation folder and asks user to choose interactively
3. Extracts results.pcl files from gradcam/, lime/, and xrai/ folders
4. Creates plots combining 3 curves per row, resulting in a 3x1 grid
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from io import BytesIO


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


def extract_result_data(base_path, method):
    """Extract results.pcl files from the XAI evaluation structure."""
    data = {}
    xai_methods = ['gradcam', 'lime', 'xrai']
    subfolder_order = ['average', 'default', 'black']  # Order for grid placement
    
    for xai_method in xai_methods:
        data[xai_method] = {}
        xai_path = os.path.join(base_path, xai_method)
        
        if not os.path.exists(xai_path):
            print(f"Warning: {xai_method} folder not found at {xai_path}")
            continue
            
        # Get all subfolders under the XAI method folder
        subfolders = get_available_folders(xai_path)
        
        for subfolder in subfolders:
            result_path = os.path.join(xai_path, subfolder, 'test', method, 'results.pcl')
            
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'rb') as f:
                        pcl_data = pickle.load(f)
                    data[xai_method][subfolder] = pcl_data
                    print(f"Loaded: {result_path}")
                except Exception as e:
                    print(f"Error loading {result_path}: {e}")
            else:
                print(f"Warning: results.pcl not found at {result_path}")
    
    return data


def create_combined_plot(data_row, xai_method, output_path):
    """Create a single plot combining data from 3 subfolders for one XAI method."""
    subfolder_order = ['average', 'default', 'black']
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(10, 6))
    
    for i, subfolder in enumerate(subfolder_order):
        if subfolder in data_row:
            pcl_data = data_row[subfolder]
            
            try:
                # Follow the same pattern as visualize_evaluation function
                if isinstance(pcl_data, dict) and xai_method in pcl_data:
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
                            
                            plt.plot(x_data, y_data, color=colors[i], label=subfolder.upper(), linewidth=2, marker='o')
                            
                            # Add annotations like in visualize_evaluation
                            for j in range(len(y_data)):
                                plt.annotate(f'{y_data[j]:.3f}',
                                           xy=(x_data[j], y_data[j]),
                                           xytext=(3, 6),
                                           textcoords='offset points',
                                           ha='center',
                                           fontsize=8)
                            
                            print(f"Successfully plotted {xai_method}/{subfolder} with {len(y_data)} points")
                        else:
                            print(f"Warning: 'correct' data is not array-like for {xai_method}/{subfolder}")
                            continue
                    else:
                        print(f"Warning: Method data is not tuple/list for {xai_method}/{subfolder}")
                        continue
                else:
                    print(f"Warning: Could not find {xai_method} key in data for {subfolder}")
                    continue
                    
            except Exception as e:
                print(f"Error plotting {xai_method}/{subfolder}: {e}")
                continue
        else:
            print(f"Warning: Missing data for {xai_method}/{subfolder}")
    
    plt.title(f'{xai_method.upper()} Results', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save plot to BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert to PIL Image
    plot_image = Image.open(buffer)
    plt.close()  # Close the figure to free memory
    
    return plot_image


def create_3x1_grid(data, output_path, spacing=30):
    """Create a 3x1 grid of combined plots."""
    xai_methods = ['gradcam', 'lime', 'xrai']
    
    # Create individual plots for each XAI method
    plot_images = []
    
    for xai_method in xai_methods:
        if xai_method in data:
            plot_img = create_combined_plot(data[xai_method], xai_method, output_path)
            plot_images.append(plot_img)
            print(f"Created plot for {xai_method}")
        else:
            print(f"Warning: No data found for {xai_method}")
            # Create empty placeholder
            plot_images.append(Image.new('RGB', (800, 480), 'white'))
    
    if not plot_images:
        print("Error: No valid plots created")
        return False
    
    # Calculate grid dimensions
    max_width = max(img.width for img in plot_images)
    max_height = max(img.height for img in plot_images)
    
    grid_width = max_width + 2 * spacing
    grid_height = 3 * max_height + 4 * spacing  # 3 rows + 4 spacing areas
    
    # Create white background
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Place plots in grid (3 rows, 1 column)
    for row, plot_img in enumerate(plot_images):
        y = spacing + row * (max_height + spacing)
        x = spacing + (max_width - plot_img.width) // 2  # Center horizontally
        
        grid_image.paste(plot_img, (x, y))
        print(f"Placed {xai_methods[row]} plot at position ({x}, {y})")
    
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
    
    # Steps 4-7: Extract data from gradcam, lime, and xrai folders
    data = extract_result_data(eval_folder_path, args.method)
    
    # Verify we have some data
    total_data_files = sum(len(method_data) for method_data in data.values())
    if total_data_files == 0:
        print("Error: No results.pcl files found in the expected structure")
        return 1
    
    print(f"Found {total_data_files} data files total")
    
    # Step 8: Create 3x1 grid of combined plots
    success = create_3x1_grid(data, args.output)
    
    if success:
        print(f"Successfully created grid: {args.output}")
        return 0
    else:
        print("Failed to create grid")
        return 1


if __name__ == '__main__':
    sys.exit(main())
