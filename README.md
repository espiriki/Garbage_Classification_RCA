# Offical Git Repo for the thesis "What Goes Where In Calgary? A Garbage Classification System Based on Images and Natural Language"

## Instructions on how to repo the results in the paper
First, download the dataset [here](https://zenodo.org/records/15832061), the zip file is named "final_dataset_20k.zip".
Unzip the folders **Final_dataset_W2025_Train**, **Final_dataset_W2025_Val**, **Final_dataset_W2025_Test**. They contain the train set, validation set and test set.

## Set up environment

Use the following singularity image to run the commands with apptainer: here.
The Dockerfile used to create this apptainer image is on the root of this repo and is named **Dockerfile**.


### Training Results

To repro the train results, use the slurm files located in the **slurm_files** dir. The python scripts in there are called with all the hyperparamters passed in the command line. The hyperparameters which are ommited are called with their default values, defined in the file **options.py** in this repo.

### Test Set Results

The train scripts will save a .pth file with the weights of the model whenever the highest validation accuracy is reached. Use the .pth file with the highest validation accuracy as the input to the test scripts.

The following commands will create an image with the confusion matrix and a .csv file with the test report in the folder **test_set_reports**.

The generic format to test an image model is:

```
python calculate_test_accuracy_image.py --image_model=<model_arch> --model_path=<path_to_the_pth_weights_file> --dataset_folder_name=./Final_dataset_W2025_Test/
```

The valid options for the **image_model** param are "eff_v2_small", "eff_v2_medium", "eff_v2_large", "convnext", "shuffle_net", "transformer_B16" and "transformer_L16"

The generic format to test a text model is:

```
python calculate_test_accuracy_text.py --text_model=<model_arch> --model_path=<path_to_the_pth_weights_file> --dataset_folder_name=./Final_dataset_W2025_Test/
```

The valid options for the **image_model** param are "distilbert", "roberta", "bert", "mobile_bert", "gpt2"

The generic format to test a multimodal model is:

```
python calculate_test_accuracy_both.py --late_fusion=MM_RCA --model_path=<path_to_the_pth_weights_file> --dataset_folder_name=./Final_dataset_W2025_Test/ --reverse
```
If the **late_fusion** parameter is changed to "hierarchical", the multimodal model with the hierarchical late fusion strategy will be used.

If the **late_fusion** is kept as MM_RCA, the following command line combinations can be added to test the different late fusion stragies:

    Adding the --features-only: simple concat
    Adding the --cross_attention_only: only RCA output
    Removing the --reverse: cross attention + simple concat

Keeping the command as-is will test the MM-RCA model.
    
    




