#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --partition=alpha
#SBATCH --job-name=Your_job_name
#SBATCH --mail-user=your.mail@tu-dresden.de
# simple_cnn 1200  simple_cnn 1500 resnet50 1500
module purge
module load release/24.04 GCCcore/12.2.0 Python/3.10.8
source /home/h3/luch715g/XAIprj/XAIenv/bin/activate
# Model number meaning: 25 for font size, 120 for alpha value, 255255255 for the RGB value of the watermark text
xaiev train --architecture simple_cnn --max_epochs 100 --model_number 25120255255255  --learning_rate 1e-3 --random_seed_train 1300 --base-dir /data/horse/ws/luch715g-XAI_workspace/data/geometry_512_25_120
