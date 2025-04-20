#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from torchvision import transforms
import torchvision
from models import *
from options import args_parser
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch
import matplotlib.pyplot as plt
import math
import csv
import keep_aspect_ratio
import albumentations as A
import cv2
import albumentations.pytorch as a_pytorch
import numpy as np
import wandb
import torch.nn as nn
import itertools
import time
from CVPR_code.CustomImageTextFolder import *
from torchmetrics.classification import ConfusionMatrix
import ssl
from sklearn.metrics import classification_report
from datetime    import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pytz
from pathlib import Path

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

BASE_PATH = os.path.dirname(os.path.realpath(__file__)) + os.sep
TRAIN_DATASET_PATH = "Train"
VAL_DATASET_PATH = "Val"


def get_class_weights(train_dataset_path):

    train_set = CustomImageTextFolder(train_dataset_path)

    total_num_samples_dataset = 0.0
    num_samples_each_class = []
    for i in range(_num_classes):
        num_samples_each_class.append(len(train_set.per_class[i]))
        total_num_samples_dataset += (len(train_set.per_class[i]))

    class_weights = []

    for i in range(_num_classes):
        class_weight = total_num_samples_dataset / \
            (_num_classes * num_samples_each_class[i])
        class_weights.append(class_weight)

    return class_weights


def run_one_epoch(epoch_num, model, data_loader, len_train_data, hw_device,
                  batch_size, train_optimizer, weights, use_class_weights, acc_steps, smoothing):

    batch_loss = []
    n_batches = math.ceil((len_train_data/batch_size))

    opt_weights = torch.FloatTensor(weights).cuda()

    if use_class_weights is True:
        criterion = torch.nn.CrossEntropyLoss(weight=opt_weights, label_smoothing=smoothing).to(hw_device)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing).to(hw_device)

    print("Using device: {}".format(hw_device))
    for batch_idx, (data, labels) in enumerate(data_loader):
        images = data['image']['raw_image']

        images, labels = images.to(hw_device), labels.to(hw_device)

        model_outputs = model(images)
        loss = criterion(model_outputs, labels)

        loss.backward()
        if acc_steps != 0:
            loss = loss / acc_steps

            if ((batch_idx + 1) % acc_steps == 0) or \
                    (batch_idx + 1 == len(data_loader)) or acc_steps == 0:
                # Update Optimizer
                print("Optimizer step on batch idx: {}".format(batch_idx))
                train_optimizer.step()
                train_optimizer.zero_grad()
        else:
            train_optimizer.step()
            train_optimizer.zero_grad()

        print("Batches {}/{} on epoch {}".format(batch_idx,
                                                 n_batches, epoch_num), end='\r')

        cpu_loss = loss.cpu()
        cpu_loss = cpu_loss.detach()
        batch_loss.append(cpu_loss)

    print("\n")

    return n_batches, batch_loss


def flatten(l):
    return [item for sublist in l for item in sublist]


def calculate_set_accuracy(
        model,
        data_loader,
        len_data,
        device,
        batch_size):

    n_batches = math.ceil((len_data/batch_size))

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):
            images = data['image']['raw_image']
            images, labels = images.to(
                device), labels.to(device)

            # Inference
            outputs = model(images)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx,
                                          n_batches), end='\r')

            all_labels.append(labels.cpu())
            all_predictions.append(pred_labels.cpu())

        all_labels = flatten(all_labels)
        all_predictions = flatten(all_predictions)

        report = classification_report(all_labels, all_predictions,
                                       target_names=["black", "blue", "green", "ttr"], output_dict=True)

        print(report)

        acc = 100 * (correct/len_data)
        print("Set acc: ", acc)
        return acc, report


def save_model_weights(model, model_name, epoch_num, val_acc, hw_device, fine_tuning, class_weights, opt):

    base = os.path.join("model_weights", model_name)
    Path(os.path.join(BASE_PATH,base)).mkdir(parents=True, exist_ok=True)    

    if fine_tuning:
        filename = "BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_Opt_{}_FractionLR_{}_VAL_ACC_{:.5f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, opt, args.fraction_lr, val_acc)

    else:

        filename = "BEST_model_{}_epoch_{}_LR_{}_Reg_{}_Opt_{}_VAL_ACC_{:.5f}_".format(
            model_name, epoch_num+1, args.lr, args.reg, opt, val_acc)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = filename + current_time
    full_path = os.path.join(BASE_PATH,base,filename)
    full_path = full_path + ".pth"

    print("Saving weights to {}".format(full_path))

    model.to("cpu")
    torch.save(model.state_dict(), full_path)

    model.to(hw_device)


def calculate_mean_std_train_dataset(train_dataset_path, pipeline):

    stats_train_data = CustomImageTextFolder(root=train_dataset_path,
                                             transform=Transforms(img_transf=pipeline))

    stats_loader = torch.utils.data.DataLoader(dataset=stats_train_data,
                                               batch_size=128,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)

    channels_sum, std_sum, num_batches = 0, 0, 0

    for images, _ in stats_loader:
        images = images['image']['raw_image']

        channels_sum += torch.mean(images*1.0, dim=[0, 2, 3])
        std_sum += torch.std(images*1.0, dim=[0, 2, 3])/len(images)
        num_batches += 1

    mean = (channels_sum / num_batches)/255
    std = (std_sum / num_batches)/255

    return mean, std


def count_parameters(model): return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    args = args_parser()

    ssl._create_default_https_context = ssl._create_unverified_context

    if not torch.cuda.is_available():
        print("GPU not available!!!!")
    else:
        print("GPU OK!!!")

    if args.tl is True:
        print("In Transfer Learning mode!!!")

    if args.dataset_folder_name == "":
        print("Please provide dataset path")
        sys.exit(1)

    # torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Image Model: {}".format(args.image_model))

    global_model = EffNetB4(_num_classes, args.tl)
    input_size = eff_net_sizes["b4"]
    _batch_size = 256
    _batch_size_FT = 42

    model = args.image_model
    # 4 012 672 parameters
    if model == "b0":
        global_model = EffNetB0(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 256
        _batch_size_FT = 256
    # 17 555 788 parameters
    elif model == "b4":
        global_model = EffNetB4(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 256
        _batch_size_FT = 42
    # 17 555 788 parameters
    elif model == "b5":
        global_model = EffNetB5(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 128
        _batch_size_FT = 16
        args.acc_steps = 2
    # 20 182 612 parameters
    elif model == "eff_v2_small":
        global_model = EffNetV2_S(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 96
        _batch_size_FT = 64
    # 52 863 480 parameters
    elif model == "eff_v2_medium":
        global_model = EffNetV2_M(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 128
        _batch_size_FT = 24
        args.acc_steps = 12
    # 117 239 396 parameters
    elif model == "eff_v2_large":
        global_model = EffNetV2_L(_num_classes, args.tl)
        input_size = eff_net_sizes[model]
        _batch_size = 96
        _batch_size_FT = 12
        args.acc_steps = 6
    # 11 178 564 parameters
    elif model == "res18":
        global_model = ResNet18(_num_classes, args.tl)
        input_size = (448, 448)
        _batch_size = 128
        _batch_size_FT = 224
    # 23 516 228 parameters
    elif model == "res50":
        global_model = ResNet50(_num_classes, args.tl)
        input_size = (448, 448)
        _batch_size = 64
        _batch_size_FT = 80
    # 58 152 004 parameters
    elif model == "res152":
        global_model = ResNet152(_num_classes, args.tl)
        input_size = (448, 448)
        _batch_size = 64
        _batch_size_FT = 38
        args.acc_steps = 2
    # 4 207 156 parameters
    elif model == "mb":
        global_model = MBNetLarge(_num_classes, args.tl)
        input_size = (320, 320)
        _batch_size = 256
        _batch_size_FT = 256
    elif model == "convnext":
        global_model = ConvNextBase(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
        _batch_size_FT = 120
        args.acc_steps = 0
    elif model == "transformer_B16":
        global_model = VisionB16(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
        _batch_size_FT = 200
    elif model == "transformer_L16":
        global_model = VisionL16(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 128
        _batch_size_FT = 52
        args.acc_steps = 3
    elif model == "shuffle_net":
        global_model = shuffleNet(_num_classes, args.tl)
        input_size = (224, 224)
        _batch_size = 256
        _batch_size_FT = 128
        args.acc_steps = 3
    else:
        print("Invalid Model: {}".format(model))
        sys.exit(1)

    print("Num total parameters of the model: {}".format(
        count_parameters(global_model)))
    print("Batch Size: {}".format(_batch_size))
    print("Batch Size FT: {}".format(_batch_size_FT))
    print("Learning Rate: {}".format(args.lr))
    print("Regularization Rate: {}".format(args.reg))
    print("Using class weights: {}".format(args.balance_weights))
    print("Optimizer: {}".format(args.opt))
    print("Grad Acc steps: {}".format(args.acc_steps))

    print("Training for {} epochs".format(args.epochs))
    if args.tl is True:
        print("Training for {} fine tuning epochs".format(args.ft_epochs))
        print("Fraction of the LR for fine tuning: {}".format(args.fraction_lr))

    config = dict(
        num_model_parameters=count_parameters(global_model),
        batch_size=_batch_size,
        batch_size_FT=_batch_size_FT,
        learning_rate=args.lr,
        regularization=args.reg,
        balance_weights=args.balance_weights,
        optimizer=args.opt,
        batch_acc_steps=args.acc_steps,
        num_epochs=args.epochs,
        fine_tuning_epochs=args.ft_epochs,
        fraction_lr=args.fraction_lr,
        architecture=args.image_model,
        dataset_id="garbage",
        prob_augmentations=args.prob_aug,
    )

    timezone = pytz.timezone('America/Edmonton')
    now = datetime.now(timezone)
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    run = wandb.init(
        project="Garbage Classification Image - Dataset v2",
        config=config,
        name="Image model: " + str(args.image_model) + " " + str(date_time)
    )

    wandb.watch(global_model)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

    WIDTH = input_size[0]
    HEIGHT = input_size[1]
    AR_INPUT = WIDTH / HEIGHT

    STATS_PIPELINE = A.Compose([
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_LINEAR),
        a_pytorch.transforms.ToTensorV2()
    ])

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_dataset_path = os.path.join(BASE_PATH, dataset_folder)

    class_weights = get_class_weights(train_dataset_path)

    print("Class weights: {}".format(class_weights))

    if args.calculate_dataset_stats is True:
        print("Calculating Train Dataset statistics...")
        mean_train_dataset, std_train_dataset = calculate_mean_std_train_dataset(
            train_dataset_path, STATS_PIPELINE)
    else:
        # ImageNet mean and std        
        mean_train_dataset = [0.485, 0.456, 0.406]
        std_train_dataset = [0.229, 0.224, 0.225]

    print("Mean Train Dataset: {}, STD Train Dataset: {}".format(
        mean_train_dataset, std_train_dataset))

    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset, always_apply=True)

    prob_augmentations = args.prob_aug

    TRAIN_PIPELINE = A.Compose([
        A.Rotate(p=prob_augmentations, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT,
                 value=0, crop_border=True),
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_LINEAR),
        A.GaussianBlur(p=prob_augmentations),
        A.VerticalFlip(p=prob_augmentations),
        A.HorizontalFlip(p=prob_augmentations), 
        A.RandomBrightnessContrast(p=prob_augmentations),
        A.Sharpen(p=prob_augmentations),
        A.Perspective(p=prob_augmentations,
                      pad_mode=cv2.BORDER_CONSTANT,
                      pad_val=0),
        # Using this transform just to zoom in an out
        A.ShiftScaleRotate(shift_limit=0, rotate_limit=0,
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT,
                           value=0, p=prob_augmentations,
                           scale_limit=0.5),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    VALIDATION_PIPELINE = A.Compose([
        keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
        A.Resize(width=WIDTH,
                 height=HEIGHT,
                 interpolation=cv2.INTER_LINEAR),
        normalize_transform,
        a_pytorch.transforms.ToTensorV2()
    ])

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    train_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        transform=Transforms(img_transf=TRAIN_PIPELINE))

    aux = [args.dataset_folder_name, VAL_DATASET_PATH]
    dataset_folder = '_'.join(aux)
    val_data = CustomImageTextFolder(
        root=os.path.join(BASE_PATH, dataset_folder),
        transform=Transforms(img_transf=VALIDATION_PIPELINE))

    _num_workers = 16

    print(train_data.class_to_idx)

    data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=_batch_size,
                                                    shuffle=True,
                                                    num_workers=_num_workers,
                                                    pin_memory=True)

    data_loader_val = torch.utils.data.DataLoader(dataset=val_data,
                                                  batch_size=_batch_size,
                                                  shuffle=True,
                                                  num_workers=_num_workers,
                                                  pin_memory=True)

    data_loader_train_FT = torch.utils.data.DataLoader(dataset=train_data,
                                                       batch_size=_batch_size_FT,
                                                       shuffle=True,
                                                       num_workers=_num_workers,
                                                       pin_memory=True)

    data_loader_val_FT = torch.utils.data.DataLoader(dataset=val_data,
                                                     batch_size=_batch_size_FT,
                                                     shuffle=True,
                                                     num_workers=_num_workers,
                                                     pin_memory=True)

    print(f"Total num of train images: {len(train_data)}")
    for i in range(_num_classes):
        len_samples = len(train_data.per_class[i])
        print("Num of samples for class {}: {}. Percentage of dataset: {:.2f}".format(
            i, len_samples, (len_samples/len(train_data))*100))

    train_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    if args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            global_model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
            global_model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        print("Invalid optimizer!")
        sys.exit(1)

    print("Starting training...")
    print("Len of train set:", len(data_loader_train.dataset))
    print("Len of val set:", len(data_loader_val.dataset))
    global_model.to(device)
    max_val_accuracy = 0.0
    best_epoch = 0
    scheduler = ReduceLROnPlateau(optimizer, 'max',factor=0.2,verbose=True)

    for epoch in range(args.epochs):

        global_model.train()
        st = time.time()

        num_batches, train_loss_per_batch = run_one_epoch(epoch,
                                                          global_model,
                                                          data_loader_train,
                                                          len(data_loader_train.dataset),
                                                          device,
                                                          _batch_size,
                                                          optimizer,
                                                          class_weights,
                                                          args.balance_weights,
                                                          args.acc_steps,
                                                          args.label_smoothing)

        elapsed_time = time.time() - st
        print('Epoch time: {:.1f}'.format(elapsed_time))

        train_loss_avg = np.average(train_loss_per_batch)
        train_loss_history.append(train_loss_avg)

        print("Avg train loss on epoch {}: {:.3f}".format(epoch, train_loss_avg))
        print("Max train loss on epoch {}: {:.3f}".format(
            epoch, np.max(train_loss_per_batch)))
        print("Min train loss on epoch {}: {:.3f}".format(
            epoch, np.min(train_loss_per_batch)))

        global_model.eval()

        print("Starting train accuracy calculation for epoch {}".format(epoch))
        train_accuracy, _ = calculate_set_accuracy(global_model,
                                                   data_loader_train,
                                                   len(data_loader_train.dataset),
                                                   device,
                                                   _batch_size)

        print("Train set accuracy on epoch {}: {:.3f} ".format(
            epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting val accuracy calculation for epoch {}".format(epoch))
        val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                          data_loader_val,
                                                          len(data_loader_val.dataset),
                                                          device,
                                                          _batch_size)

        print("Val set accuracy on epoch {}: {:.3f}".format(
            epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        wandb.log({'epoch': epoch,
                   'epoch_time_seconds': elapsed_time,
                   'train_loss_avg': train_loss_avg,
                   'train_accuracy_history': train_accuracy,
                   'val_accuracy_history': val_accuracy,
                   'max_val_acc': max_val_accuracy,
                   'black_val_precision': val_report["black"]["precision"],
                   'blue_val_precision': val_report["blue"]["precision"],
                   'green_val_precision': val_report["green"]["precision"],
                   'ttr_val_precision': val_report["ttr"]["precision"]})

        if val_accuracy > max_val_accuracy:
            print("Best model obtained based on Val Acc. Saving it!")
            save_model_weights(global_model, args.image_model,
                               epoch, val_accuracy, device, False, args.balance_weights, args.opt)
            max_val_accuracy = val_accuracy
            best_epoch = epoch
        else:
            print("Not saving model on epoch {}, best Val Acc so far on epoch {}: {:.3f}".format(epoch, best_epoch,
                                                                                                 max_val_accuracy))

    print("Starting Fine tuning!!")
    # Fine tuning loop
    if args.tl is True:

        # set all model parameters to train
        for param in global_model.parameters():
            param.requires_grad = True

        # update learning rate of optimizer
        for group in optimizer.param_groups:
            group['lr'] = args.lr/args.fraction_lr

        for epoch in range(args.ft_epochs):

            global_model.train()
            st = time.time()
            # train using a small learning rate
            _, ft_train_loss_per_batch = run_one_epoch(epoch,
                                                       global_model,
                                                       data_loader_train_FT,
                                                       len(train_data),
                                                       device,
                                                       _batch_size_FT,
                                                       optimizer,
                                                       class_weights,
                                                       args.balance_weights,
                                                       args.acc_steps,
                                                       args.label_smoothing)
            elapsed_time = time.time() - st
            print('Fine Tuning: epoch time: {:.1f}'.format(elapsed_time))

            train_loss_avg = np.average(ft_train_loss_per_batch)

            print("Fine Tuning: avg train loss on epoch {}: {:.3f}".format(
                epoch, train_loss_avg))
            print("Fine Tuning: max train loss on epoch {}: {:.3f}".format(
                epoch, np.max(ft_train_loss_per_batch)))
            print("Fine Tuning: min train loss on epoch {}: {:.3f}".format(
                epoch, np.min(ft_train_loss_per_batch)))

            train_loss_history.append(train_loss_avg)
            global_model.eval()

            print(
                "Fine Tuning: starting train accuracy calculation for epoch {}".format(epoch))
            train_accuracy, _ = calculate_set_accuracy(global_model,
                                                       data_loader_train_FT,
                                                       len(train_data),
                                                       device,
                                                       _batch_size)

            print("Fine Tuning: train set accuracy on epoch {}: {:.3f} ".format(
                epoch, train_accuracy))
            train_accuracy_history.append(train_accuracy)

            print(
                "Fine Tuning: starting validation accuracy calculation for epoch {}".format(epoch))
            val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                              data_loader_val_FT,
                                                              len(val_data),
                                                              device,
                                                              _batch_size)

            print("Fine Tuning: Val set accuracy on epoch {}: {:.3f}".format(
                epoch, val_accuracy))

            scheduler.step(val_accuracy)
            val_accuracy_history.append(val_accuracy)

            if val_accuracy > max_val_accuracy:
                print("Fine Tuning: best model obtained based on Val Acc. Saving it!")
                save_model_weights(global_model, args.image_model,
                                   epoch, val_accuracy, device, True, args.balance_weights, args.opt)
                best_epoch = epoch
                max_val_accuracy = val_accuracy
            else:
                print("Fine Tuning: not saving model, best Val Acc so far on epoch {}: {:.3f}".format(best_epoch,
                                                                                                      max_val_accuracy))
                
            wandb.log({'epoch': epoch,
                       'epoch_time_seconds': elapsed_time,
                       'train_loss_avg': train_loss_avg,
                       'train_accuracy_history': train_accuracy,
                       'val_accuracy_history': val_accuracy,
                       'max_val_acc': max_val_accuracy,
                       'black_val_precision': val_report["black"]["precision"],
                       'blue_val_precision': val_report["blue"]["precision"],
                       'green_val_precision': val_report["green"]["precision"],
                       'ttr_val_precision': val_report["ttr"]["precision"]})                

    run.finish()