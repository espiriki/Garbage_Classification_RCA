#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torchvision.models import *
import torch.nn as nn


def EffNetV2_L(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_v2_l(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)

    return model


def EffNetV2_M(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_v2_m(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)

    return model


def EffNetV2_S(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_v2_s(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)

    return model


def EffNetB4(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_b4(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)

    return model


def EffNetB5(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_b5(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def EffNetB0(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = efficientnet_b0(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def ResNet18(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = resnet18(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                         out_features=num_classes)
    return model


def ResNet50(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = resnet50(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                         out_features=num_classes)
    return model


def ResNet152(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = resnet152(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                         out_features=num_classes)
    return model


def ConvNextBase(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = convnext_base(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def MBNetLarge(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = mobilenet_v3_large(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)
    return model


def VisionB16(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = vit_b_16(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features=_in_features,
                                 out_features=num_classes)

    return model

def VisionL16(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = vit_l_16(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features=_in_features,
                                 out_features=num_classes)

    return model


def ShuffleNetV2(num_classes=4, transfer_learning=False):

    if transfer_learning:
        _weights = 'IMAGENET1K_V1'
    else:
        _weights = None

    model = shufflenet_v2_x2_0(weights=_weights)

    if transfer_learning:
        for param in model.parameters():
            param.requires_grad = False

    _in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=_in_features,
                                    out_features=num_classes)

    return model