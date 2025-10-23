# The procedure for investigating "random seed influence comparison

1. Training the model
Use the following command to train models. The parameter **brake** is applied here so that training stops when the accuracy reaches 0.9. 
This is applied for a fair comparison between models. 

**Note that is advised to set the model number the same as the random seed to make the seed appear in the name of the model.**
```
xaiev train --architecture alexnet_simple --max_epochs 100 --model_number 1000 --learning_rate 1e-4 --random_seed_train 1000 --brake
```


2. Creating saliency maps
Use the following command to create saliency maps. Do this for all the models with different random seeds and apply all XAI methods you want.
```
xaiev create-saliency-maps --xai-method gradcam --model alexnet_simple_2000_30  
```


3. Creating evaluation images
Use the following command to create evaluation images for all the corresponding saliency maps. There are 3 options for masks: average, black and original. When trying to use the original background, the parameter "patch" should be dismissed.
```
xaiev create-eval-images --xai-method gradcam --model simple_cnn_1000_70 --patch average
```


4. Getting evaluation results
Use the following command to get the result. Do this for all the corresponding evaluation images you have and with both revelation and occlusion.
```
xaiev eval --xai-method gradcam --model simple_cnn_1500_63 --eval-method revelation --patch average
```


5. Combining curves to compare
Use the script **compare_seed_results_combine_curves.py** to create images that combine the result of different seeds of models and plot the curves in one figure. Change the **base_folder** in the main function to your base address and add your model type to  **model_options**. Run the script to get the plots.

**Note that the script can only combine the result of two models with different seeds. You need to change it if you want to compare more models in one plot.**

**The script is written with the help of Aider and it might not be 100% faultproof under all circumstances.**