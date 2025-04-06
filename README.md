# Offical Git Repo for the paper "Multimodal Garbage Classification Using Reverse Cross Attention"

## Instructions on how to repo the results in the paper
First, download the dataset [here](https://zenodo.org/uploads/10215592). Unzip the folders **CVPR_2024_dataset_Train**, **CVPR_2024_dataset_Val**, **CVPR_2024_dataset_Test** it into the root folder of this repo.

Then, to repro the results in the test set, download the saved model weights from the previous link and extract the *.pth files into the folder **saved_model_weights** of this repo.

## Set up environment

Use the following docker image to run the commands: `jcazarin/cazarin_pytorch_cvpr:2024`. It is publicly available on DockerHub.

### Training Results

To repro the training process of the **EfficientNet B0**, use the following command:

    python main_image.py --image_model=b0 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=70 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet B4**, use the following command:

    python main_image.py--dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --image_model=b4 --epochs=30 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet B5**, use the following command:

    python main_image.py --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --image_model=b5 --epochs=40 --balance_weights --reg=0.1 

To repro the training process of the **EfficientNet V2 Small**, use the following command:

    python main_image.py --image_model=eff_v2_small --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **EfficientNet V2 Medium**, use the following command:

    python main_image.py --image_model=eff_v2_medium -dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=40 --balance_weights --reg=0.1 --prob_aug=0.9

To repro the training process of the **EfficientNet V2 Large**, use the following command:

    python main_image.py --image_model=eff_v2_large --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=40 --balance_weights --reg=0.1

To repro the training process of the **ResNet 18**, use the following command:

    python main_image.py --image_model=res18 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=40 --opt=adamw --balance_weights --reg=0.1

To repro the training process of the **ResNet 50**, use the following command:

    python main_image.py --image_model=res50 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=40 --opt=adamw --balance_weights --reg=0.1

To repro the training process of the **ResNet 152**, use the following command:

    python main_image.py --image_model=res152 --dataset_folder_name=CVPR_2024_dataset --epochs=50 --ft_epochs=60 --opt=adamw --balance_weights --reg=0.1 --prob_aug=0.6

To repro the training process of the **MobileNet V3-large**, use the following command:

    python main_image.py --image_model=mb --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=50 --balance_weights --reg=0.1

To repro the training process of the **ConvNeXt Base**, use the following command:

    python main_image.py --image_model=convnext --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **ViT-B/16**, use the following command:

    python main_image.py --image_model=transformer_B16 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **ViT-L/16**, use the following command:

    python main_image.py --image_model=transformer_L16 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=40 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --prob_aug=0.8

To repro the training process of the **Distilbert**, use the following command:

    python main_text.py --text_model=distilbert --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=45 --balance_weights --reg=0.1

To repro the training process of the **RoBERTa**, use the following command:

    python main_text.py --text_model=roberta --dataset_folder_name=CVPR_2024_dataset --ft_epochs=60 --opt=adamw --epochs=60 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the **BERT**, use the following command:

 python main_text.py --text_model=bert --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the **BART**, use the following command:

    python main_text.py --text_model=bart --dataset_folder_name=CVPR_2024_dataset --ft_epochs=30 --opt=adamw --epochs=10 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the **GPT2**, use the following command:

    python main_text.py --text_model=gpt2 --dataset_folder_name=CVPR_2024_dataset --ft_epochs=50 --opt=adamw --epochs=50 --balance_weights --reg=0.1 --lr=0.0005

To repro the training process of the multi-modal **MM-RCA**, use the following command:

    python main_both.py --dataset_folder_name=CVPR_2024_dataset ft_epochs=40 --fraction_lr=3 --image_prob_dropout=0.2 --late_fusion=MMF --lr=0.0016 --image_text_dropout=0.0 --epochs=60 --opt=sgd --reg=0.00 --balance_weights --acc_steps=10 --acc_steps_FT=10 --prob_aug=1.00 --model_dropout=0.6 --num_neurons_FC=256 --text_model=distilbert --heads=16 --reverse


### Test Set Results

The following commands will create an image with the confusion matrix and a .csv file with the test report in the folder **test_set_reports**.

To repro the results in the test set of the **EfficientNet B0** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=b0 --model_path=./saved_model_weights/BEST_model_b0.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **EfficientNet B4** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=b4 --model_path=./saved_model_weights/BEST_model_b4.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **EfficientNet B5** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=b5 --model_path=./saved_model_weights/BEST_model_b5.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **EfficientNet V2 Small** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=eff_v2_small --model_path=./saved_model_weights/BEST_model_eff_v2_small.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **EfficientNet V2 Medium** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=eff_v2_medium --model_path=./saved_model_weights/BEST_model_eff_v2_medium.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **EfficientNet V2 Large** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=eff_v2_large --model_path=./saved_model_weights/BEST_model_eff_v2_large.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ResNet 18** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=res18 --model_path=./saved_model_weights/BEST_model_res18.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ResNet 50** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=res50 --model_path=./saved_model_weights/BEST_model_res50.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ResNet 152** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=res152 --model_path=./saved_model_weights/BEST_model_res152.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **MobileNet V3-large** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=mb --model_path=./saved_model_weights/BEST_model_mobilenet.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ConvNeXt Base** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=convnext --model_path=./saved_model_weights/BEST_model_convnext.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ViT-B/16** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=transformer_B16 --model_path=./saved_model_weights/BEST_model_transformer_B16.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **ViT-L/16** model , use the following command:

    python calculate_test_accuracy_image.py --image_model=transformer_L16 --model_path=./saved_model_weights/BEST_model_transformer_L16.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **Distilbert** model , use the following command:

    python calculate_test_accuracy_text.py --text_model=distilbert --model_path=./saved_model_weights/BEST_model_distilbert.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **RoBERTa** model , use the following command:

    python calculate_test_accuracy_text.py --text_model=roberta --model_path=./saved_model_weights/BEST_model_roberta.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **BERT** model , use the following command:

    python calculate_test_accuracy_text.py --text_model=bert --model_path=./saved_model_weights/BEST_model_bert.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **BART** model , use the following command:

    python calculate_test_accuracy_text.py --text_model=bart --model_path=./saved_model_weights/BEST_model_bart.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **GPT2** model , use the following command:

    python calculate_test_accuracy_text.py --text_model=gpt2 --model_path=./saved_model_weights/BEST_model_gpt2.pth --dataset_folder_name=./CVPR_2024_dataset_Test/

To repro the results in the test set of the **MM-RCA** model , use the following command:

    python calculate_test_accuracy_both.py --model_path=./saved_model_weights/BEST_model_eff_v2_medium_plus_distilbert.pth --dataset_folder_name=./CVPR_2024_dataset_Test/ --image_prob_dropout=0.0 --image_text_dropout=0.00
