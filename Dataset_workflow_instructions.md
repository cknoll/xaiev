# Instructions for using the data set generator and applying XAI tool kit

## 

## Step1: Getting the file

 Pull the **image_generator.py** from GitHub. 



## Step2: Create the dataset

 Open a terminal at the position of **image_generator.py** and run



```bash
 python image_generator.py
```



 This should create **imgs_background and imgs_mian** at the same position as the .py file



## Step3: Install all necessary packages

Follow the instruction in the Xaiev tool guide. Use

```bash
pip install -e .
```

to install all necessary packages. Run 

```bash
xaiev --bootstrap
```

in the base project file and **modify the .env file** to the folder containing imgs_main and imgs_background (e.g. XAIEV_BASE_DIR="D:/XAI/xaiev/data/geometry") 



## Step4: Train the model

Open a terminal in your project folder(the same position as the .env file) and run 

```bash
xaiev train --architecture alexnet_simple --max_epochs 60 --model_number 1 --learning_rate 1e-3
```

 to train the model. 



## Step5: Create heatmaps

Run

```bash
xaiev create-saliency-maps --xai-method gradcam --model alexnet_simple_1_60
```

to get the heatmaps.



## Step6: Create evaluation images

Run

```bash
xaiev create-eval_images --xai-method gradcam --model alexnet_simple_1_60
```



## Step7: Generate XAI evaluation results

Run

```bash
xaiev eval --xai-method gradcam --model alexnet_simple_1_60
```

and

```bash
xaiev eval --xai-method gradcam --model alexnet_simple_1_60 --eval-method occlusion
```



## Additional Note:

You should have cv package for image_generator.py. 

For Xaiev tools, check requirement.txt for all required packages.

