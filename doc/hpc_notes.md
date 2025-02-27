# HPC Notes

This file collects examples which can be used to run the xaiev pipeline on a high performance computing (HPC) node.
The commands are specific to to TU Dresden HPC but might be useful for others as well:

```bash
#!/bin/bash

#SBATCH --job-name=create-eval-images-all         # Job name
#SBATCH --output=create-eval-images-all.log      # Output log file
#SBATCH --error=create-eval-imgages-all_error.log        # Error log file
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000
#SBATCH --time=24:00:00                      # Maximum run time (e.g., 2 hours)
#SBATCH --partition=alpha                    # Partition to run on
#SBATCH --ntasks=1
#SBATCH --mail-type=start,end
#SBATCH --mail-user=firstname.lastname@tu-dresden.de


# Load necessary modules
module purge
module load release/24.04 GCCcore/12.2.0 Python/3.10.8 CUDA/11.7

# Activate your virtual environment
source ~/xaivenv/bin/activate

# Run the Python script from base dir
cd /data/horse/ws/xaiev-workdir

xaiev create-eval-images --xai-method gradcam --model simple_cnn_1_1
xaiev create-eval-images --xai-method gradcam --model resnet50_1_1
xaiev create-eval-images --xai-method gradcam --model convnext_tiny_1_1
xaiev create-eval-images --xai-method gradcam --model vgg16_1_1

xaiev create-eval-images --xai-method lime --model simple_cnn_1_1
xaiev create-eval-images --xai-method lime --model resnet50_1_1
xaiev create-eval-images --xai-method lime --model convnext_tiny_1_1
xaiev create-eval-images --xai-method lime --model vgg16_1_1

xaiev create-eval-images --xai-method prism --model simple_cnn_1_1
xaiev create-eval-images --xai-method prism --model resnet50_1_1
xaiev create-eval-images --xai-method prism --model convnext_tiny_1_1
xaiev create-eval-images --xai-method prism --model vgg16_1_1

xaiev create-eval-images --xai-method xrai --model simple_cnn_1_1
xaiev create-eval-images --xai-method xrai --model resnet50_1_1
xaiev create-eval-images --xai-method xrai --model convnext_tiny_1_1
xaiev create-eval-images --xai-method xrai --model vgg16_1_1

```