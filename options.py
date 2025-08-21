#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100,
                        help="number of rounds of training")
    parser.add_argument('--dataset_folder_name', type=str, default="",
                        help="dataset folder name in the base location")
    parser.add_argument('--dataset_folder_name_val', type=str, default="",
                        help="val dataset folder name in the base location")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--image_text_dropout', type=float, default=0.33,
                        help='change of dropping either text or image')
    parser.add_argument('--image_prob_dropout', type=float, default=0.7,
                        help='change of dropping image when dropping the modalities')    
    parser.add_argument('--reg', type=float, default=1e-2,
                        help='regularization rate')
    parser.add_argument('--model_dropout', type=float, default=0.6,
                        help='model FC layer dropout')
    parser.add_argument(
        '--tl', action=argparse.BooleanOptionalAction, default=True, help="Whether to use transfer learning or not")
    parser.add_argument('--balance_weights',
                        action=argparse.BooleanOptionalAction, default=False, help="Whether to use class balance weights or not")
    parser.add_argument('--ft_epochs', type=int, default=15,
                        help='number of fine tuning epochs')
    parser.add_argument('--fraction_lr', type=float, default=5,
                        help='value to divide the regular LR for to use in fine tuning')
    parser.add_argument('--image_model', type=str, default='b4', help='model name')
    parser.add_argument('--text_model', type=str, default='distilbert', help='model name')

    parser.add_argument('--model_path', type=str, default="",
                        help='Model file to calculate accuracy against the test set. Must match the model architecture select with the -model parameter')

    parser.add_argument('--acc_steps', type=int, default=0,
                        help='Gradient accumulation steps')
    
    parser.add_argument('--acc_steps_FT', type=int, default=0,
                        help='Gradient accumulation steps')

    parser.add_argument('--num_neurons_FC', type=int, default=256,
                        help='Num neurons in FC layers')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')

    parser.add_argument('--opt', type=str, default="sgd",
                        help='Optimizer to use')

    parser.add_argument('--base_path',
                        type=str, default="D:\Mestrado\ENSF_619_02_Final_project_jose_cazarin\BEST_MODELS_CVPR_2025",
                        help='base_path')

    parser.add_argument('--calculate_dataset_stats',
                        action=argparse.BooleanOptionalAction, default=False, help="Calculate the development set stats used for normalization")
    
    parser.add_argument('--prob_aug', type=float, default=0.6,
                        help='Probability of applying augmentations')

    parser.add_argument('--late_fusion', type=str, default="gated",
                        help='Which late fusion strategy to use')

    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Fraction to use Label Smoothing')

    parser.add_argument('--name', type=str,
                        help='Run description')

    parser.add_argument('--reverse',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use RCA or not")

    args = parser.parse_args()
    return args
