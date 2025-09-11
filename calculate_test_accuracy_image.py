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


eff_net_sizes = {
    'b0': (256, 224),
    'b4': (384, 380),
    'b5': (489, 456),
    'b6': (561, 528),
    'b7': (633, 600),
    'eff_v2_small': (384, 384),
    'eff_v2_medium': (480, 480),
    'eff_v2_large': (480, 480)
}

BASE_PATH = "./test_set_reports"

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
    
    global_model.eval()    
    with torch.no_grad():

        for batch_idx, (data, labels) in enumerate(data_loader):

            images = data['image']['raw_image']
            images, labels = images.to(hw_device), labels.to(hw_device)

            # Inference
            outputs = model(images)

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
    Path(os.path.join(BASE_PATH,args.image_model)).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(BASE_PATH,args.image_model,
                     'conf_matrix_image_model_{}_test_set_acc_{:.2f}.png'.format(
                         args.image_model, test_acc)))

    report = classification_report(torch.tensor(all_labels).cpu(),
                                   torch.tensor(all_preds).cpu(),
                                   target_names=classes)

    report_dict = classification_report(torch.tensor(all_labels).cpu(),
                                        torch.tensor(all_preds).cpu(),
                                        target_names=classes, output_dict=True)

    dataframe = pd.DataFrame.from_dict(report_dict)
    dataframe.to_csv(os.path.join(BASE_PATH,args.image_model,
                                  "image_model_{}_report_test_set_acc_{:.2f}.csv".format(args.image_model, test_acc)),
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

    if args.image_model == "b0":
        global_model = EffNetB0(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 40
    elif args.image_model == "b4":
        global_model = EffNetB4(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 32
    elif args.image_model == "b5":
        global_model = EffNetB5(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 16        
    elif args.image_model == "eff_v2_small":
        global_model = EffNetV2_S(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 48
    elif args.image_model == "eff_v2_medium":
        global_model = EffNetV2_M(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 32
    elif args.image_model == "eff_v2_large":
        global_model = EffNetV2_L(_num_classes, args.tl)
        input_size = eff_net_sizes[args.image_model]
        _batch_size = 8
    elif args.image_model == "res18":
        global_model = ResNet18(_num_classes, args.tl)
        input_size = (300, 300)
        _batch_size = 256
    elif args.image_model == "res50":
        global_model = ResNet50(_num_classes, args.tl)
        input_size = (400, 400)
        _batch_size = 96
    elif args.image_model == "res152":
        global_model = ResNet152(_num_classes, args.tl)
        input_size = (500, 500)
        _batch_size = 32
    elif args.image_model == "mb":
        global_model = MBNetLarge(_num_classes, args.tl)
        input_size = (320, 320)
        _batch_size = 256
    elif args.image_model == "convnext":
        global_model = ConvNextBase(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 256
    elif args.image_model == "transformer":
        global_model = VisionB16(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 256
    elif args.image_model == "transformer_B16":
        global_model = VisionB16(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
    elif args.image_model == "transformer_L16":
        global_model = VisionL16(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
        args.acc_steps = 3
    elif args.image_model == "shuffle_net":
        global_model = ShuffleNetV2(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
        args.acc_steps = 3        
    else:
        print("Invalid Model: {}".format(args.model))
        sys.exit(1)

    print("Image Model: {}".format(args.image_model))

    model_name = args.model_path

    global_model.load_state_dict(torch.load(model_name))

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
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

    test_data = CustomImageTextFolder(root=args.dataset_folder_name,
                                                 transform=Transforms(img_transf=TEST_PIPELINE))

    print("Num of test images: {}".format(len(test_data)))

    _num_workers = 8

    data_loader_test = torch.utils.data.DataLoader(dataset=test_data,
                                                   batch_size=_batch_size,
                                                   shuffle=False, num_workers=_num_workers, pin_memory=True)

    test_accuracy, test_report = calculate_test_accuracy(global_model,
                                                         data_loader_test,
                                                         len(test_data),
                                                         device,
                                                         _batch_size, args)

    print(test_data.class_to_idx)
    print("Test accuracy: {:.2f} %".format(test_accuracy))
    print("Test Report:")
    print(test_report)
