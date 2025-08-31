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
from CVPR_code.text_models import *
from CVPR_code.multimodal_model import *
from torchmetrics.classification import ConfusionMatrix
import ssl
from sklearn.metrics import classification_report
from datetime import datetime
import os
import pytz
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau

_num_classes = 4

BASE_PATH = os.path.dirname(os.path.realpath(__file__)) + os.sep
TRAIN_DATASET_PATH = "Train"
VAL_DATASET_PATH = "Val"

mode_config_dict = {
    'image_only': {"remove_text": True, "remove_image": False},
    'text_only': {"remove_text": False, "remove_image": True},
    'both': {"remove_text": False, "remove_image": False}
}


class Transforms:
    def __init__(self, img_transf: A.Compose):
        self.img_transf = img_transf

    def __call__(self, img, *args, **kwargs):
        img = np.array(img)
        augmented = self.img_transf(image=img)
        image = augmented["image"]
        return image


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
        criterion = torch.nn.CrossEntropyLoss(weight=opt_weights,label_smoothing=smoothing).to(hw_device)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=smoothing).to(hw_device)

    print("Using device: {}".format(hw_device))
    for batch_idx, (data, labels) in enumerate(data_loader):
        images = data['image']['raw_image']
        texts = data['text']

        input_token_ids = texts['tokens'].to(device)
        attention_mask = texts['attention_mask'].to(device)
        images, labels = images.to(
            device), labels.to(device)

        # Inference
        model_outputs = model(_input_ids=input_token_ids,
                              _attention_mask=attention_mask,
                              _images=images)

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

        print("Batch {}/{} on epoch {}".format(batch_idx, n_batches, epoch_num))

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
        batch_size,
        mode,
        eval_mode):

    n_batches = math.ceil((len_data/batch_size))

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        correct = 0
        for batch_idx, (data, labels) in enumerate(data_loader):

            images = data['image']['raw_image']

            texts = data['text']
            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)

            images = images.to(device)
            labels = labels.to(device)

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

            correct += torch.sum(torch.eq(pred_labels, labels)).item()

            print("Batches {}/{} ".format(batch_idx, n_batches))

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


def save_model_weights(model, text_model_name, image_model_name,\
    epoch_num, val_acc, hw_device, fine_tuning, class_weights, opt, fusion):

    base = os.path.join("model_weights", text_model_name+"_"+image_model_name)
    Path(os.path.join(BASE_PATH,base)).mkdir(parents=True, exist_ok=True)    

    if fine_tuning:
        filename = "BEST_model_{}_FT_EPOCH_{}_LR_{}_Reg_{}_FractionLR_{}_OPT_{}_VAL_ACC_{:.5f}".format(
            text_model_name+"_"+image_model_name, epoch_num+1, args.lr, args.reg, args.fraction_lr, opt, val_acc)

    else:

        filename = "BEST_model_{}_epoch_{}_LR_{}_Reg_{}_VAL_ACC_{:.5f}_".format(
            text_model_name+"_"+image_model_name, epoch_num+1, args.lr, args.reg, val_acc)

    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = filename + "_" + fusion + "_"
    filename = filename + current_time
    full_path = os.path.join(BASE_PATH,base,filename)
    full_path = full_path + ".pth"

    print("Saving weights to {}".format(full_path))
    model.to("cpu")
    torch.save(model.state_dict(), full_path)

    model.to(hw_device)


def count_parameters(model): return sum(p.numel() for p in model.parameters())


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]


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

    # # This is to make results predictable between runs
    # torch.manual_seed(42)
    # np.random.seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.image_model = "EffNetv2-Medium"

    print("Text Model: {}".format(args.text_model))
    print("Image Model: {}".format(args.image_model))

    global_model = None
    if args.text_model == "bart":
        _batch_size = 32
        _batch_size_FT = 2
    else:
        _batch_size = args.batch_size
        _batch_size_FT = args.batch_size_FT

    if args.late_fusion == "gated":
        global_model = EffV2MediumAndDistilbertGated(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model)
    elif args.late_fusion == "classic":
        global_model = EffV2MediumAndDistilbertClassic(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model)
    elif args.late_fusion == "normalized":
        global_model = EffV2MediumAndDistilbertNormalized(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model,
            _batch_size)
    elif args.late_fusion == "clip":
        global_model = EffV2MediumAndDistilbertCLIP(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model,
            _batch_size)
    elif args.late_fusion == "MM_RCA":
        global_model = MM_RCA(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model,
            _batch_size,
            args.reverse,
            args.features_only)
    elif args.late_fusion == "hierarchical":
        global_model = Hierarchical(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model,
            _batch_size,
            args.reverse)
    elif args.late_fusion == "bimodal":
        _batch_size = 32
        _batch_size_FT = 4
        
        global_model = HierarchicalBimodalFusion(
            _num_classes,
            args.model_dropout,
            args.image_text_dropout,
            args.image_prob_dropout,
            args.num_neurons_FC,
            args.text_model,
            _batch_size,
            args.reverse)
    else:
        print("Wrong late fusion strategy: ", args.late_fusion)
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
    print("Grad Acc steps for Fine Tuning: {}".format(args.acc_steps_FT))
    print("Img dropout prob: {}".format(args.image_prob_dropout))
    print("Modality dropout prob: {}".format(args.image_text_dropout))
    print("Late Fusion strategy: {}".format(args.late_fusion))
    print("Num neurons FC: {}".format(args.num_neurons_FC))
    print("Prob Img Aug: {}".format(args.prob_aug))

    print("Training for {} epochs".format(args.epochs))
    if args.tl is True:
        print("Training for {} fine tuning epochs".format(args.ft_epochs))
        print("Fraction of the LR for fine tuning: {}".format(args.fraction_lr))

    config = dict(args)

    timezone = pytz.timezone('America/Edmonton')
    now = datetime.now(timezone)
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    print("Starting W&B...")
    run = wandb.init(
        project="Garbage Classification Both - Dataset v3 - better fusion - comparison",
        config=config,
        # name="Both models: " +
        # str(args.text_model) +
        # str(args.image_model) +
        # " " + str(date_time) +
        # " " + str(args.late_fusion)
        name="HIERARCHICAL FUSION: " + str(args.late_fusion) + " " + str(date_time)
    )
    print("Done!")

    wandb.watch(global_model)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        global_model = nn.DataParallel(global_model)

    # EffNetV2 Medium image size
    size = global_model.get_image_size()
    WIDTH = size[0]
    HEIGHT = size[1]
    AR_INPUT = WIDTH / HEIGHT

    _tokenizer = global_model.get_tokenizer()
    _max_len = global_model.get_max_token_size()

    # Imagenet mean and std
    mean_train_dataset = [0.485, 0.456, 0.406]
    std_train_dataset = [0.229, 0.224, 0.225]

    prob_augmentations = args.prob_aug
    normalize_transform = A.Normalize(mean=mean_train_dataset,
                                      std=std_train_dataset, always_apply=True)

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
    print(aux)
    dataset_folder = '_'.join(aux)
    print(dataset_folder)
    train_dataset_path = \
        os.path.join(BASE_PATH, dataset_folder)
    print(train_dataset_path)        

    class_weights = get_class_weights(train_dataset_path)
    print("Class weights: {}".format(class_weights))

    _tokenizer = global_model.get_tokenizer()
    _max_len = global_model.get_max_token_size()

    aux = [args.dataset_folder_name, TRAIN_DATASET_PATH]
    train_dataset_folder = '_'.join(aux)
    train_dataset_folder = os.path.join(BASE_PATH, train_dataset_folder)
    print("Train dataset folder:", train_dataset_folder)
    train_data = CustomImageTextFolder(
        root=train_dataset_folder,
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=TRAIN_PIPELINE))

    aux = [args.dataset_folder_name, VAL_DATASET_PATH]
    val_dataset_folder = '_'.join(aux)
    val_dataset_folder = os.path.join(BASE_PATH, val_dataset_folder)
    print("Val dataset folder:", val_dataset_folder)
    val_data = CustomImageTextFolder(
        root=val_dataset_folder,
        tokens_max_len=_max_len,
        tokenizer_text=_tokenizer,
        transform=Transforms(img_transf=VALIDATION_PIPELINE))

    _num_workers = 16

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

    print(f"Total num of samples: {len(train_data)}")
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
    global_model.to(device)
    max_val_accuracy = 0.0
    max_img_only_acc = 0.0
    max_txt_only_acc = 0.0
    best_epoch = 0
    scheduler = ReduceLROnPlateau(optimizer, 'max',factor=0.4,verbose=True)

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
        eval_mode = False
        # Calculating the train set accuracy with the drop out of images or text
        train_accuracy, _ = calculate_set_accuracy(global_model,
                                                   data_loader_train,
                                                   len(data_loader_train.dataset),
                                                   device,
                                                   _batch_size,
                                                   # this mode is not used when eval_mode is False
                                                   mode_config_dict['both'],
                                                   eval_mode)

        print("Train set accuracy with model dropout on epoch {}: {:.3f} ".format(
            epoch, train_accuracy))
        train_accuracy_history.append(train_accuracy)

        print("Starting val accuracy calculation for epoch {}".format(epoch))
        eval_mode = False
        # Calculating the val set accuracy with the drop out of images or text
        val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                          data_loader_val,
                                                          len(data_loader_val.dataset),
                                                          device,
                                                          _batch_size,
                                                          # this mode is not used when eval_mode is False
                                                          mode_config_dict['both'],
                                                          eval_mode)

        print("Val set accuracy with model dropout on epoch {}: {:.3f} ".format(
            epoch, val_accuracy))
        val_accuracy_history.append(val_accuracy)

        if val_accuracy > max_val_accuracy:
            print("Best model obtained based on Val Acc. Saving it!")
            save_model_weights(global_model, args.text_model, args.image_model,
                               epoch, val_accuracy, device, False, args.balance_weights, args.opt, args.late_fusion)
            max_val_accuracy = val_accuracy
            best_epoch = epoch
        else:
            print("Not saving model on epoch {}, best Val Acc so far on epoch {}: {:.3f}".format(epoch, best_epoch,
                                                                                                 max_val_accuracy))

        eval_mode = True
        print("Calculating val set accuracy with image only on epoch {}".format(epoch))
        val_acc_image_only, _ = calculate_set_accuracy(global_model,
                                                       data_loader_val,
                                                       len(data_loader_val.dataset),
                                                       device,
                                                       _batch_size,
                                                       mode_config_dict['image_only'],
                                                       eval_mode)

        print("Calculating val set accuracy with text only on epoch {}".format(epoch))
        val_acc_text_only, _ = calculate_set_accuracy(global_model,
                                                      data_loader_val,
                                                      len(data_loader_val.dataset),
                                                      device,
                                                      _batch_size,
                                                      mode_config_dict['text_only'],
                                                      eval_mode)
        
        # print("Calculating val set accuracy with BOTH on epoch {}".format(epoch))
        # val_acc_both, _ = calculate_set_accuracy(global_model,
        #                                               data_loader_val,
        #                                               len(data_loader_val.dataset),
        #                                               device,
        #                                               _batch_size,
        #                                               mode_config_dict['both'],
        #                                               eval_mode)        

        if val_acc_image_only > max_img_only_acc:
            max_img_only_acc = val_acc_image_only

        if val_acc_text_only > max_txt_only_acc:
            max_txt_only_acc = val_acc_text_only

        wandb.log({'epoch': epoch,
                   'epoch_time_seconds': elapsed_time,
                   'train_loss_avg': train_loss_avg,
                   'train_accuracy_history': train_accuracy,
                   'val_accuracy_history': val_accuracy,
                   'val_accuracy_text_only_history': val_acc_text_only,
                   'val_accuracy_image_only_history': val_acc_image_only,
                #    'val_accuracy_BOTH_history': val_acc_both,
                   'max_val_acc': max_val_accuracy,
                   'max_img_only_val_acc': max_img_only_acc,
                   'max_txt_only_val_acc': max_txt_only_acc,
                   'black_val_precision': val_report["black"]["precision"],
                   'blue_val_precision': val_report["blue"]["precision"],
                   'green_val_precision': val_report["green"]["precision"],
                   'ttr_val_precision': val_report["ttr"]["precision"]})

    print("Starting Fine tuning!!")
    # Fine tuning loop
    if args.tl is True:

        # set all model parameters to train
        for param in global_model.text_model.parameters():
            param.requires_grad = True
        # set all model parameters to train
        for param in global_model.image_model.parameters():
            param.requires_grad = True
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
            ft_num_batches, ft_train_loss_per_batch = run_one_epoch(epoch,
                                                                    global_model,
                                                                    data_loader_train_FT,
                                                                    len(train_data),
                                                                    device,
                                                                    _batch_size_FT,
                                                                    optimizer,
                                                                    class_weights,
                                                                    args.balance_weights,
                                                                    args.acc_steps_FT,
                                                                    args.label_smoothing)
            elapsed_time = time.time() - st
            print('Fine Tuning: epoch time: {:.1f}'.format(elapsed_time))

            ft_train_loss_avg = np.average(ft_train_loss_per_batch)

            print("Fine Tuning: avg train loss on epoch {}: {:.3f}".format(
                epoch, ft_train_loss_avg))
            print("Fine Tuning: max train loss on epoch {}: {:.3f}".format(
                epoch, np.max(ft_train_loss_per_batch)))
            print("Fine Tuning: min train loss on epoch {}: {:.3f}".format(
                epoch, np.min(ft_train_loss_per_batch)))

            train_loss_history.append(ft_train_loss_avg)

            global_model.eval()
            print(
                "Fine Tuning: starting train accuracy calculation for epoch {}".format(epoch))

            eval_mode = False
            # Calculating the train set accuracy with the drop out of images or text
            train_accuracy, _ = calculate_set_accuracy(global_model,
                                                       data_loader_train_FT,
                                                       len(train_data),
                                                       device,
                                                       _batch_size_FT,
                                                       # this mode is not used when eval_mode is False
                                                       mode_config_dict['both'],
                                                       eval_mode)

            print("Fine Tuning: train set accuracy on epoch {}: {:.3f} ".format(
                epoch, train_accuracy))
            train_accuracy_history.append(train_accuracy)

            print(
                "Fine Tuning: starting val accuracy calculation for epoch {}".format(epoch))

            eval_mode = False
            # Calculating the val set accuracy with the drop out of images or text
            val_accuracy, val_report = calculate_set_accuracy(global_model,
                                                              data_loader_val_FT,
                                                              len(val_data),
                                                              device,
                                                              _batch_size_FT,
                                                              # this mode is not used when eval_mode is False
                                                              mode_config_dict['both'],
                                                              eval_mode)

            print("Fine Tuning: Val set accuracy on epoch {}: {:.3f}".format(
                epoch, val_accuracy))
            
            scheduler.step(val_accuracy)
            val_accuracy_history.append(val_accuracy)

            if val_accuracy > max_val_accuracy:
                print("Fine Tuning: best model obtained based on Val Acc. Saving it!")
                save_model_weights(global_model, args.text_model, args.image_model,
                                   epoch, val_accuracy, device, True, args.balance_weights, args.opt, args.late_fusion)
                best_epoch = epoch
                max_val_accuracy = val_accuracy
            else:
                print("Fine Tuning: not saving model, best Val Acc so far on epoch {}: {:.3f}".format(best_epoch,
                                                                                                      max_val_accuracy))

            eval_mode = True
            print(
                "Fine Tuning: calculating val set accuracy with image only on epoch {}".format(epoch))
            val_acc_image_only, _ = calculate_set_accuracy(global_model,
                                                           data_loader_val_FT,
                                                           len(val_data),
                                                           device,
                                                           _batch_size_FT,
                                                           mode_config_dict['image_only'],
                                                           eval_mode)

            print(
                "Fine Tuning: calculating val set accuracy with text only on epoch {}".format(epoch))
            val_acc_text_only, _ = calculate_set_accuracy(global_model,
                                                          data_loader_val_FT,
                                                          len(val_data),
                                                          device,
                                                          _batch_size_FT,
                                                          mode_config_dict['text_only'],
                                                          eval_mode)
            
            # print("Fine Tuning: Calculating val set accuracy with BOTH on epoch {}".format(epoch))
            # val_acc_both, _ = calculate_set_accuracy(global_model,
            #                                           data_loader_val_FT,
            #                                           len(val_data),
            #                                           device,
            #                                           _batch_size_FT,
            #                                           mode_config_dict['both'],
            #                                           eval_mode)              

            if val_acc_image_only > max_img_only_acc:
                max_img_only_acc = val_acc_image_only

            if val_acc_text_only > max_txt_only_acc:
                max_txt_only_acc = val_acc_text_only
            
            wandb.log({'epoch': epoch,
                       'epoch_time_seconds': elapsed_time,
                       'train_loss_avg': ft_train_loss_avg,
                       'train_accuracy_history': train_accuracy,
                       'val_accuracy_history': val_accuracy,
                       'val_accuracy_text_only_history': val_acc_text_only,
                       'val_accuracy_image_only_history': val_acc_image_only,
                    #    'val_accuracy_BOTH_history': val_acc_both,
                       'max_val_acc': max_val_accuracy,
                       'max_img_only_val_acc': max_img_only_acc,
                       'max_txt_only_val_acc': max_txt_only_acc,
                       'black_val_precision': val_report["black"]["precision"],
                       'blue_val_precision': val_report["blue"]["precision"],
                       'green_val_precision': val_report["green"]["precision"],
                       'ttr_val_precision': val_report["ttr"]["precision"]})

    run.finish()