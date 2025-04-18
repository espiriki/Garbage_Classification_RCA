#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from torchvision import transforms
import torchvision
from models import *
from options import args_parser
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch
import math
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
import torch.nn as nn
import keep_aspect_ratio
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report
from CVPR_code.text_models import *
from CVPR_code.CustomImageTextFolder import *
from CVPR_code.multimodal_model import *

_num_classes = 4
classes = ["Black", "Blue", "Green", "TTR"]

class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image

mode_config_dict = {
    'image_only': {"remove_text": True, "remove_image": False},
    'text_only': {"remove_text": False, "remove_image": True},
    'both': {"remove_text": False, "remove_image": False}
}

BASE_PATH = "./test_set_reports"

def calculate_test_accuracy(
        model,
        data_loader,
        len_test_data,
        hw_device,
        batch_size,
        mode,
        eval_mode):

    correct = 0
    n_batches = math.ceil((len_test_data/batch_size))
    model.to(hw_device)
    all_preds = []
    all_labels = []
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    with torch.no_grad():

        for batch_idx, (data, labels) in enumerate(data_loader):

            texts = data['text']
            images = data['image']['raw_image']

            input_token_ids = texts['tokens'].to(hw_device)
            attention_mask = texts['attention_mask'].to(hw_device)            
            images = images.to(hw_device)
            labels = labels.to(hw_device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask,
                            _images=images,
                            eval=eval_mode,
                            remove_text=mode["remove_text"],
                            remove_image=mode["remove_image"]
                            )

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            all_preds.append(pred_labels)
            all_labels.append(labels)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Test batches {}/{} ".format(batch_idx,
                                               n_batches))

            print("Running test accuracy: {:.3f} %".format(
                100*(correct/len_test_data)))

    print("\n")
    print("samples checked for test: {}".format(len_test_data))
    print("correct samples for test: {}".format(correct))
    test_acc = 100 * (correct/len_test_data)
    all_preds = [item for sublist in all_preds for item in sublist]
    all_labels = [item for sublist in all_labels for item in sublist]

    conf_matrix = confmat(torch.tensor(all_labels), torch.tensor(all_preds))
    print(conf_matrix)

    report = classification_report(torch.tensor(all_labels).cpu(),
                                   torch.tensor(all_preds).cpu(),
                                   target_names=classes)

    report_dict = classification_report(torch.tensor(all_labels).cpu(),
                                        torch.tensor(all_preds).cpu(),
                                        target_names=classes, output_dict=True)

    return test_acc, report, report_dict, conf_matrix

def generate_report_and_image(test_report_dict,test_accuracy, conf_matrix, mode):
    
        
    dataframe = pd.DataFrame.from_dict(test_report_dict)
    dataframe.to_csv(os.path.join(BASE_PATH,
                                  "multimodal_model_report_test_set_acc_{:.2f}_{}.csv".format(test_accuracy, mode)), index=True)

    df_cm = pd.DataFrame(conf_matrix, index=classes,
                         columns=classes)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 5))
    sn.heatmap(df_cm, annot=True, cmap='viridis', fmt='g')
    plt.savefig(
        os.path.join(BASE_PATH,
                     'conf_matrix_multimodal_model_test_set_acc_{:.2f}_{}.png'.format(test_accuracy, mode)))

    print(test_data.class_to_idx)
    print("Test accuracy random both: {:.2f} %".format(test_accuracy))
    print("Test Report:")
    print(test_report)

if __name__ == '__main__':
    args = args_parser()

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # This is to make results predictable between runs
    torch.manual_seed(42)
    np.random.seed(42)

    _batch_size = 16

    global_model = EffV2MediumAndDistilbertMMF(
        _num_classes,
        args.model_dropout,
        args.image_text_dropout,
        args.image_prob_dropout,
        args.num_neurons_FC,
        args.text_model,
        _batch_size) 

    model_name = args.model_path
    global_model.load_state_dict(torch.load(model_name))

    global_model.eval()

    # Eff Net V2 Medium
    WIDTH = 480
    HEIGHT = 480
    AR_INPUT = WIDTH / HEIGHT

    # ImageNet mean and std
    mean_train_dataset = [0.485, 0.456, 0.406]
    std_train_dataset = [0.229, 0.224, 0.225]

    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset, always_apply=True)

    TEST_PIPELINE = A.Compose([
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_LINEAR),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    _tokenizer = global_model.get_tokenizer()
    _max_len = global_model.get_max_token_size()

    test_data = CustomImageTextFolder(
        root=args.dataset_folder_name,
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=TEST_PIPELINE))

    print("Num of test samples: {}".format(len(test_data)))

    _num_workers = 8

    data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=_batch_size,
                                                   shuffle=True,
                                                   num_workers=_num_workers,
                                                   pin_memory=True)

    test_accuracy, test_report, test_report_dict, conf_matrix = calculate_test_accuracy(global_model,
                                                            data_loader_test,
                                                            len(test_data),
                                                            device,
                                                            _batch_size,
                                                            # will always use both inputs
                                                            mode_config_dict['both'],
                                                            True)
    
    generate_report_and_image(test_report_dict,test_accuracy, conf_matrix, "always_both")