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
# import wandb
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

_num_classes = 4


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


BASE_PATH = "./test_set_reports"

def get_dummy_pipeline():
    pipeline = A.Compose([
        A.Resize(width=320,
                 height=320,
                 interpolation=cv2.INTER_CUBIC),
        a_pytorch.transforms.ToTensorV2()
    ])

    return pipeline

def calculate_test_accuracy(
        model,
        data_loader,
        len_test_data,
        hw_device,
        batch_size,
        args):

    correct = 0
    n_batches = math.ceil((len_test_data/batch_size))
    model.to(hw_device)
    all_preds = []
    all_labels = []
    confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    with torch.no_grad():

        for batch_idx, (data, labels) in enumerate(data_loader):

            texts = data['text']

            input_token_ids = texts['tokens'].to(hw_device)
            attention_mask = texts['attention_mask'].to(hw_device)
            labels = labels.to(hw_device)

            # Inference
            outputs = model(_input_ids=input_token_ids,
                            _attention_mask=attention_mask)

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

    classes = ["Black", "Blue", "Green", "TTR"]

    df_cm = pd.DataFrame(conf_matrix, index=classes,
                         columns=classes)

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 5))
    sn.heatmap(df_cm, annot=True, cmap='viridis', fmt='g')
    Path(os.path.join(BASE_PATH,args.text_model)).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(BASE_PATH,args.text_model,
                     'conf_matrix_text_model_{}_class_weights_{}_test_set_acc_{:.2f}.png'.format(
                         args.text_model, args.balance_weights, test_acc)))

    report = classification_report(torch.tensor(all_labels).cpu(),
                                   torch.tensor(all_preds).cpu(),
                                   target_names=classes)

    report_dict = classification_report(torch.tensor(all_labels).cpu(),
                                        torch.tensor(all_preds).cpu(),
                                        target_names=classes, output_dict=True)

    dataframe = pd.DataFrame.from_dict(report_dict)
    dataframe.to_csv(os.path.join(BASE_PATH,args.text_model,
                                  "text_model_{}_report_test_set_acc_{:.2f}.csv".format(args.text_model, test_acc)),
                     index=True)

    return test_acc, report


if __name__ == '__main__':
    args = args_parser()

    if args.model_path == "":
        print("Please provide test model path")
        sys.exit(0)

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 66 365 956 parameters
    if args.text_model == "distilbert":
        global_model = DistilBert(_num_classes, args.model_dropout)
        _batch_size = 128
    # 124 648 708 parameters
    elif args.text_model == "roberta":
        global_model = Roberta(_num_classes, args.model_dropout)
        _batch_size = 128
    # 109 485 316 parameters
    elif args.text_model == "bert":
        global_model = Bert(_num_classes, args.model_dropout)
        _batch_size = 128
    # 407 345 156 parameters
    elif args.text_model == "bart":
        global_model = Bart(_num_classes, args.model_dropout)
        _batch_size = 4
    # 124 442 884 parameters
    elif args.text_model == "gpt2":
        global_model = GPT2(_num_classes)
        _batch_size = 32
    elif args.text_model == "mobile_bert":
        global_model = MobileBERT(_num_classes)
        _batch_size = 18
        args.acc_steps = 12
    else:
        print("Invalid Model: {}".format(args.text_model))
        sys.exit(1)

    print("Text Model: {}".format(args.text_model))

    model_name = args.model_path

    global_model.load_state_dict(torch.load(model_name))

    global_model.eval()

    _tokenizer = global_model.get_tokenizer()
    _max_len = global_model.get_max_token_size()

    test_data = CustomImageTextFolder(
        root=args.dataset_folder_name,
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=get_dummy_pipeline()))

    print("Num of test texts: {}".format(len(test_data)))

    _num_workers = 8

    data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=_batch_size,
                                                   shuffle=True,
                                                   num_workers=_num_workers,
                                                   pin_memory=True)

    if "true" in args.model_path or "True" in args.model_path:
        args.balance_weights = True

    if "false" in args.model_path or "False" in args.model_path:
        args.balance_weights = False

    test_accuracy, test_report = calculate_test_accuracy(global_model,
                                                         data_loader_test,
                                                         len(test_data),
                                                         device,
                                                         _batch_size, args)

    print(test_data.class_to_idx)
    print("Test accuracy: {:.2f} %".format(test_accuracy))
    print("Test Report:")
    print(test_report)
