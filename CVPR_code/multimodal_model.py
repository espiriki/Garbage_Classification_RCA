import torch
from transformers import DistilBertModel, DistilBertConfig, BartConfig, BartForSequenceClassification
from transformers import BartModel
from transformers import BertModel, BertConfig
from torchvision.models import *
from transformers import DistilBertTokenizer, BartTokenizer
from transformers import BertTokenizer
import numpy as np
import sys
import torch.nn.functional as F
import random


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, reverse, attn_dropout=0.1):
        super().__init__()
        self.reverse = reverse
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        softmax_output = F.softmax(attn, dim=-1)

        if self.reverse:
            print("RCA!!!")
            dimension = softmax_output.shape[-1]
            softmax_output = (1.0-softmax_output)/(dimension-1)
        
        # --------------------------------------------------------------------------
        print("weights shape:", softmax_output.shape)
        X = softmax_output.shape[-1]
        # Flatten the first three dimensions into one for easy indexing
        flat_tensor = softmax_output.view(-1, X)  # shape: [(A*B*C), X]

        # Randomly choose indices to sample
        num_samples = 2
        random_indices = random.sample(range(flat_tensor.size(0)), num_samples)
        
        print(f"Sampling {num_samples} entries from weights tensor with shape \
            {flat_tensor.shape} based on last dimension {X}:")
        print("len(random_indices): ", len(random_indices))
        print("Size dimension: ", X)
        
        truncate_length = 25
        for idx in random_indices:
            full_slice = flat_tensor[idx]
            print(f"Sample {idx}: {full_slice[:truncate_length]} ...")
            
        # --------------------------------------------------------------------------            

        attn = self.dropout(softmax_output)
        output = torch.matmul(attn, v)

        return output, attn


class SelfAttention(torch.nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = torch.nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5, reverse=False)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, _ = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q


class CrossAttention(torch.nn.Module):
    def __init__(self, n_head, d_model_x1, d_model_x2, dk, dv,relu, reverse=False):
        super().__init__()

        self.n_head = n_head
        self.dk = dk
        self.dv = dv
        self.relu = relu
        self.w_qs = torch.nn.Linear(d_model_x1, n_head * dk, bias=False)
        self.w_ks = torch.nn.Linear(d_model_x2, n_head * dk, bias=False)
        self.w_vs = torch.nn.Linear(d_model_x2, n_head * dv, bias=False)
        self.fc = torch.nn.Linear(n_head * dv, d_model_x2, bias=False)

        self.residual_proj = torch.nn.Identity()
        if d_model_x1 != n_head * dv:
            self.residual_proj = torch.nn.Linear(d_model_x1, n_head * dv)

        self.attention = \
            ScaledDotProductAttention(temperature=dk ** 0.5, reverse=reverse)

        self.reverse = reverse
        self.first_time = False

        self.dropout = torch.nn.Dropout(0.1)
        self.norm = torch.nn.LayerNorm(d_model_x2)
        self.relu_layer = torch.nn.ReLU()

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.dk, self.dv, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, _ = self.attention(q, k, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        residual = self.residual_proj(residual)

        q += residual

        q = self.norm(q)

        if self.relu:
            print("Using RELU")
            q = self.relu_layer(q)

        return q

def decision(probability):
    return np.random.rand(1)[0] < probability

def eff_net_v2():

    model = efficientnet_v2_m(weights='IMAGENET1K_V1')

    # Remove classifier and pooling layers
    # remove avgpool and classifier
    model = torch.nn.Sequential(*list(model.children())[:-2])

    for param in model.parameters():
        param.requires_grad = False

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


class EffV2MediumAndDistilbertGated(torch.nn.Module):

    def __init__(self,
                 n_classes,
                 drop_ratio,
                 image_or_text_dropout_chance,
                 img_prob_dropout,
                 num_neurons_fc,
                 text_model_name,
                 batch_size,
                 num_heads,
                 relu,
                 reverse=False):
        super(EffV2MediumAndDistilbertGated, self).__init__()

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
        self.fc_layer_neurons = num_neurons_fc

        self.image_dropout = torch.nn.Dropout2d(p=1.0)
        self.text_dropout = torch.nn.Dropout1d(p=1.0)
        self.image_or_text_dropout_chance = image_or_text_dropout_chance
        self.img_dropout_prob = img_prob_dropout

        # 1280 from image + 768 from text
        self.image_to_hidden_size = \
            torch.nn.Linear(1280,
                      out_features=self.fc_layer_neurons)

        self.text_to_hidden_size = \
            torch.nn.Linear(in_features=self.text_model.config.hidden_size,
                      out_features=self.fc_layer_neurons)

        self.concat_layer = \
            torch.nn.Linear(self.fc_layer_neurons*2, self.fc_layer_neurons)

        # FC layer to classes
        self.fc_layer = \
            torch.nn.Linear(self.fc_layer_neurons, n_classes)

        # Layers for gated output
        self.gated_output_hidden_size = 256
        self.hyper_tang_layer = torch.nn.Tanh()
        self.softmax_layer = torch.nn.Softmax(dim=1)

        self.image_features_hidden_layer = \
            torch.nn.Linear(1280,
                      self.gated_output_hidden_size)

        self.text_features_hidden_layer = \
            torch.nn.Linear(self.text_model.config.hidden_size,
                      self.gated_output_hidden_size)

        self.z_layer = \
            torch.nn.Linear(self.gated_output_hidden_size * 2,
                      self.gated_output_hidden_size)

        # FC layer to classes
        self.fc_layer_gated = \
            torch.nn.Linear(self.gated_output_hidden_size, n_classes)

        # FC layer to classes
        self.clip_fc_layer = torch.nn.Linear(batch_size, n_classes)
        self.batch_size = batch_size

        self.trans_conv = torch.nn.ConvTranspose1d(
            in_channels=8, out_channels=8, kernel_size=2, stride=2, padding=0, output_padding=0)

        # 0.07 is the temperature parameter
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * np.log(1 / 0.07))

        self.output_all_features = torch.nn.Linear(640, 4)

        # d_model
        input_size_txt = 768
        input_size_img = 1280

        print("input size text: ", input_size_txt)
        print("input size image: ", input_size_img)

        # h (num heads)
        self.num_heads = num_heads

        print("Num heads: ", self.num_heads)
        print("RELU: ", relu)

        dk = 64
        dv = 64

        # d_model, dk, dv
        self.self_attention_text = SelfAttention(
            self.num_heads,
            input_size_txt,
            dk,
            dv)

        self.self_attention_image = SelfAttention(
            self.num_heads,
            input_size_img,
            dk,
            dv)

        dv_cross_t_to_i = int(input_size_img / self.num_heads)
        # d_model_txt, d_model_img, dk, dv
        print("Cross Attention reverse: ", reverse)

        print("dv_cross_t_to_i: ", dv_cross_t_to_i)

        self.cross_attention_T_to_I = CrossAttention(
            self.num_heads,
            input_size_txt,
            input_size_img,
            dk,
            dv_cross_t_to_i,
            reverse,
            relu)

        dv_cross_i_to_t = int(input_size_txt / self.num_heads)

        print("dv_cross_i_to_t: ", dv_cross_i_to_t)

        # d_model_img, d_model_txt, dk, dv
        self.cross_attention_I_to_T = CrossAttention(
            self.num_heads,
            input_size_img,
            input_size_txt,
            dk,
            dv_cross_i_to_t,
            reverse,
            relu)

        # self.final = torch.nn.Linear(
        #     (cross_attention_output_size_txt +
        #         cross_attention_output_size_img)*self.num_heads, n_classes)

        self.final_features_only = torch.nn.Linear(
            1280+768, n_classes)
        
        # self.final_with_everything = torch.nn.Linear(
        #     (input_size_img +
        #         input_size_txt)*self.num_heads +
        #     1280+768, n_classes)

        self.final_with_everything = torch.nn.Linear(
            (input_size_img +
                input_size_txt)*2, n_classes)


    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Gated forward pass")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)
        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        
        text_features = text_output[0][:, 0]
        image_features = self.image_model(self._images)
        print("text_features:",text_features.shape)
        print("image_features:",image_features.shape)

        # 256 * bs
        image_feats_after_tanh =\
            self.hyper_tang_layer(
                self.image_features_hidden_layer(image_features))
        # 256 * bs
        text_feats_after_tanh =\
            self.hyper_tang_layer(
                self.text_features_hidden_layer(text_features))
        # print("image_feats_after_tanh shape: ", image_feats_after_tanh.shape)
        # print("text_feats_after_tanh shape: ", text_feats_after_tanh.shape)

        # 512 * bs
        concat_output_before_tanh = torch.cat(
            (self.image_features_hidden_layer(image_features),
             self.text_features_hidden_layer(text_features)), dim=1)
        # print("concat_output_before_tanh shape: ", concat_output_before_tanh.shape)

        # in 512*bs and out 256 * bs
        z_layer_output = self.softmax_layer(
            self.z_layer(concat_output_before_tanh))
        # print("z_layer_output shape: ", z_layer_output.shape)

        # z_images will be 256 * bs
        z_images = z_layer_output * image_feats_after_tanh
        # print("z_images shape: ", z_images.shape)

        # z_texts will be 256 * bs
        z_texts = (1 - z_layer_output) * text_feats_after_tanh
        # print("z_texts shape: ", z_texts.shape)

        gate_output = z_images + z_texts
        # print("gate_output shape: ", gate_output.shape)

        after_dropout = self.drop(gate_output)

        final_output = self.fc_layer_gated(after_dropout)
        # print("final_output shape: ", final_output.shape)

        return final_output

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

    def self_attention_block(self, x):
        keys = x.matmul(self.W_key)
        queries = x.matmul(self.W_query)
        values = x.matmul(self.W_value)

        # unnormalized attention weights
        attn_scores = queries.matmul(keys.T)

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vec = attn_weights.matmul(values)
        print("output shape of self attention:", context_vec.shape)
        return context_vec

    def cross_attention_block(self, x_1, x_2):
        queries_1 = x_1.matmul(self.W_query_cross)
        keys_2 = x_2.matmul(self.W_key_cross)
        values_2 = x_2.matmul(self.W_value_cross)

        attn_scores = queries_1.matmul(keys_2.T)
        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vec = attn_weights.matmul(values_2)
        print("output shape of cross attention:", context_vec.shape)
        return context_vec



class EffV2MediumAndDistilbertClassic(EffV2MediumAndDistilbertGated):
    
    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Classic forward")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]
        text_features = hidden_state[:, 0]

        image_features = self.image_model(self._images)

        image_hidden_size = self.image_to_hidden_size(image_features)
        text_hidden_size = self.text_to_hidden_size(text_features)

        image_plus_text_features = torch.cat(
            (image_hidden_size, text_hidden_size), dim=1)

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)

        return final_output
    
    
class EffV2MediumAndDistilbertNormalized(EffV2MediumAndDistilbertGated):
    
    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("Normalized forward")
        # print("forward pass shape images: ", _images.shape)
        # print("forward pass shape text: ", _input_ids.shape)

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]
        
        text_features = hidden_state[:, 0]
        image_features = self.image_model(self._images)

        image_hidden_size = self.image_to_hidden_size(image_features)
        text_hidden_size = self.text_to_hidden_size(text_features)

        image_hidden_size = image_hidden_size / image_hidden_size.norm(dim=1, keepdim=True)
        text_hidden_size = text_hidden_size / text_hidden_size.norm(dim=1, keepdim=True)

        image_plus_text_features = torch.cat(
            (image_hidden_size, text_hidden_size), dim=1)

        after_concat = self.concat_layer(image_plus_text_features)
        after_drop = self.drop(after_concat)
        final_output = self.fc_layer(after_drop)

        return final_output    



class EffV2MediumAndDistilbertCLIP(EffV2MediumAndDistilbertGated):

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        print("CLIP forward")

        # During evaluation we want to use the dropout
        # to always remove only images or always remove
        # only text

        self._images=_images
        self._input_ids=_input_ids
        self._attention_mask=_attention_mask
        self.drop_modalities(eval, remove_image, remove_text)

        text_output = self.text_model(
            input_ids=self._input_ids,
            attention_mask=self._attention_mask
        )
        hidden_state = text_output[0]

        text_features = hidden_state[:, 0]
        image_features = self.image_model(self._images)

        image_features = self.image_to_hidden_size(image_features)
        text_features = self.text_to_hidden_size(text_features)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        if logits_per_image.shape[0] != self.batch_size:
            print("using max unpool")
            logits_per_image = self.trans_conv(logits_per_image)

        # print("logits_per_image after max pool: ", logits_per_image.shape)

        final_output = self.clip_fc_layer(logits_per_image)

        return final_output


class EffV2MediumAndDistilbertMMF(EffV2MediumAndDistilbertGated):

    def forward(self,
                _input_ids,
                _attention_mask,
                _images,
                eval=False,
                remove_image=False,
                remove_text=False):

        # print("MMF forward")

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
        original_text_features = hidden_state
        original_image_features = self.image_model(self._images)

        print("original_text_features shape:",
              original_text_features.shape)

        bs, c, h, w = original_image_features.shape
        original_image_features = \
            original_image_features.view(
                bs, c, h * w).permute(0, 2, 1)
            
        print("original_image_features shape:",
              original_image_features.shape)            

        # Normalize
        original_text_features = original_text_features / \
            original_text_features.norm(dim=1, keepdim=True)
        original_image_features = original_image_features / \
            original_image_features.norm(dim=1, keepdim=True)

        # Self attention
        text_self_attention = self.self_attention_text(
            original_text_features,
            original_text_features,
            original_text_features)
        img_self_attention = self.self_attention_image(
            original_image_features,
            original_image_features,
            original_image_features)

        # Cross attention
        complementary_cross_attention_T_I = self.cross_attention_T_to_I(
            text_self_attention,
            img_self_attention,
            img_self_attention)
        complementary_cross_attention_I_T = self.cross_attention_I_to_T(
            img_self_attention,
            text_self_attention,
            text_self_attention)

        pooled_img = complementary_cross_attention_T_I.mean(
            dim=1)  # → [16, 1280]
        pooled_txt = complementary_cross_attention_I_T.mean(
            dim=1)  # → [16, 768]

        pooled_img_orig = original_image_features.mean(
            dim=1)  # → [16, 1280]
        pooled_img_txt = original_text_features.mean(
            dim=1)  # → [16, 1280]

        # FC layer to output
        concat_features = torch.cat(
            (
                pooled_img,
                pooled_txt,
                pooled_img_orig,
                pooled_img_txt
            ), dim=1)

        after_dropout = self.drop(concat_features)

        # output = self.final(after_dropout)
        # output = self.final_features_only(after_dropout)

        # print("output FEATURES shape:",after_dropout.shape)

        output = self.final_with_everything(after_dropout)

        return output
