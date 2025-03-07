import torch
from transformers import DistilBertModel, DistilBertConfig, BartConfig, BartForSequenceClassification
from transformers import BartModel
from transformers import BertModel, BertConfig
from torchvision.models import *
from transformers import DistilBertTokenizer, BartTokenizer
from transformers import BertTokenizer
import numpy as np
import sys
from CVPR_code.multi_head_attn import MultiHeadAttention

def decision(probability):
    return np.random.rand(1)[0] < probability

def eff_net_v2():

    model = efficientnet_v2_m(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        *[model.classifier[i] for i in range(1)])

    return model

def distilbert():

    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    for param in model.parameters():
        param.requires_grad = False

    return model

def bart():

    model = BartModel.from_pretrained("facebook/bart-large")

    for param in model.parameters():
        param.requires_grad = False

    return model

def bert():

    model = BertModel.from_pretrained("bert-base-uncased")

    for param in model.parameters():
        param.requires_grad = False

    return model


class MHA_RCA(torch.nn.Module):

    def __init__(self,
                 drop_ratio,
                 image_or_text_dropout_chance,
                 img_prob_dropout,
                 text_model_name,
                 n_classes,
                 reverse):
        super(MHA_RCA, self).__init__()

        print("Using MHA_RCA model!")

        self.text_model_name = text_model_name
        
        if text_model_name == "bert":     
            self.text_model = bert()
        elif text_model_name == "distilbert":     
            self.text_model = distilbert()
        elif text_model_name == "bart":     
            self.text_model = bart()
        else:
            print("Wrong text model:", text_model_name)
            sys.exit(1)
            
        self.image_model = eff_net_v2()

        self.drop = torch.nn.Dropout(p=drop_ratio)

        self.image_dropout = torch.nn.Dropout2d(p=1.0)
        self.text_dropout = torch.nn.Dropout1d(p=1.0)
        self.image_or_text_dropout_chance = image_or_text_dropout_chance
        self.img_dropout_prob = img_prob_dropout

        input_size_txt = 768  # from the model
        input_size_img = 1280  # from the model
        self.num_heads = 16  # design choice
        self.d_model = 64  # design choice
        self.d_k = self.d_v = int(self.d_model / self.num_heads)

        self.seq_len_txt = int(input_size_txt / self.d_model)
        self.seq_len_img = int(input_size_img / self.d_model)

        print("self.num_heads: ", self.num_heads)
        print("self.d_model: ", self.d_model)
        print("self.d_k and self.d_v: ", self.d_k, self.d_v)
        print("self.seq_len_txt: ", self.seq_len_txt)
        print("self.seq_len_img: ", self.seq_len_img)
        print("\n")

        # Self attention (unimodal)
        self.self_attn_multi_head_txt = \
            MultiHeadAttention(self.d_model, self.num_heads, self.d_k, self.d_k)

        self.self_attn_multi_head_img = \
            MultiHeadAttention(self.d_model, self.num_heads, self.d_k, self.d_v)
         
        # Cross attention
        self.cross_attention_1 = MultiHeadAttention(
            self.d_model, self.num_heads, self.d_k, self.d_v, reverse)
            
        self.cross_attention_2 = MultiHeadAttention(
            self.d_model, self.num_heads, self.d_k, self.d_v, reverse)

        self.cross_attn_only = torch.nn.Linear(
            input_size_img+input_size_txt, n_classes)

        self.original_features_only = torch.nn.Linear(
            input_size_img+input_size_txt, n_classes)
        
        self.cros_attn_plus_original_features = torch.nn.Linear(
            (input_size_img+input_size_txt)*2, n_classes)

    def drop_modalities(self, _eval, remove_image, remove_text):
        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text
        if _eval:
            # when evaluating, dropout is removed
            # so set it here again
            self.image_dropout.train()
            self.text_dropout.train()
            if remove_image:
                print("    Eval: zero image")
                self._images = self.image_dropout(self._images)
            if remove_text:
                print("    Eval: zero text")
                self._input_ids = self.text_dropout(self._input_ids)
                self._attention_mask = self.text_dropout(self._attention_mask)

            if not remove_image and not remove_text:
                # print("    Eval: using both")
                pass
        # Training
        else:
            # print("self.image_or_text_dropout_chance: ",
            #       self.image_or_text_dropout_chance)
            if decision(self.image_or_text_dropout_chance):
                image_or_text = decision(self.img_dropout_prob)
                if image_or_text:
                    print("    Train: zeroing image\n")
                    self._images = self.image_dropout(self._images)
                else:
                    print("    Train: zeroing text\n")
                    self._input_ids = self.text_dropout(self._input_ids)
                    self._attention_mask = self.text_dropout(self._attention_mask)
            else:
                # print("    Train: using both\n")
                pass

    def get_tokenizer(self):
        if self.text_model_name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.text_model_name == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        elif self.text_model_name == "bart":
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")            
        
        return self.tokenizer

    def get_image_size(self):
        return (480, 480)

    def get_max_token_size(self):
        if self.text_model_name == "bert":
            self.config = BertConfig().max_position_embeddings
        elif self.text_model_name == "distilbert":
            self.config = DistilBertConfig().max_position_embeddings
        elif self.text_model_name == "bart":
            self.config = BartConfig().max_position_embeddings            
        
        return self.config

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        self._images = _images
        self._input_ids = _input_ids
        self._attention_mask = _attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]

        # Get the image and text features
        original_text_features = hidden_state[:, 0]
        original_image_features = self.image_model(self._images)

        # Normalize
        original_text_features = original_text_features / \
            original_text_features.norm(dim=1, keepdim=True)
        original_image_features = original_image_features / \
            original_image_features.norm(dim=1, keepdim=True)

        bs = original_text_features.shape[0]
        # Reshape to the format required by MHA
        original_text_features_reshaped_multi_head = \
            original_text_features.view(
                bs, self.seq_len_txt, self.d_model)
            
        original_img_features_reshaped_multi_head = \
            original_image_features.view(
                bs, self.seq_len_img, self.d_model)

        # Self attention
        multi_head_output_self_attn_txt, _ = \
            self.self_attn_multi_head_txt(original_text_features_reshaped_multi_head,
            original_text_features_reshaped_multi_head,
            original_text_features_reshaped_multi_head)
            
        multi_head_output_self_attn_img, _ = \
            self.self_attn_multi_head_img(original_img_features_reshaped_multi_head,
            original_img_features_reshaped_multi_head,
            original_img_features_reshaped_multi_head)            

        # Cross attention
        complementary_cross_attention_T_I, _ = self.cross_attention_1(
            multi_head_output_self_attn_txt,
            multi_head_output_self_attn_img,
            multi_head_output_self_attn_img)
        complementary_cross_attention_I_T, _ = self.cross_attention_2(
            multi_head_output_self_attn_img,
            multi_head_output_self_attn_txt,
            multi_head_output_self_attn_txt)

        # Flatten
        complementary_cross_attention_T_I = torch.flatten(
            complementary_cross_attention_T_I, start_dim=1, end_dim=2)
        complementary_cross_attention_I_T = torch.flatten(
            complementary_cross_attention_I_T, start_dim=1, end_dim=2)

        # FC layer to output
        concat_features = torch.cat(
            (
                complementary_cross_attention_T_I,
                complementary_cross_attention_I_T,
                original_image_features,
                original_text_features
            ), dim=1)

        after_dropout = self.drop(concat_features)

        # output = self.cross_attn_only(after_dropout)
        # output = self.original_features_only(after_dropout)
        output = self.cros_attn_plus_original_features(after_dropout)

        return output
