import torch
from transformers import DistilBertModel, DistilBertConfig, BartConfig, BartForSequenceClassification
from transformers import BartModel
from transformers import BertModel, BertConfig
from torchvision.models import *
from transformers import DistilBertTokenizer, BartTokenizer
from transformers import BertTokenizer
import numpy as np
import sys

class EfficientNetV2MFullFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.stem = model.features[:2]
        self.stage1 = model.features[2]
        self.stage2 = model.features[3]
        self.stage3 = model.features[4]
        self.stage4 = model.features[5]
        self.stage5 = model.features[6]
        self.stage6 = model.features[7]
        self.final_conv = model.features[8]  # Conv-BN-ReLU
        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        out_stage3 = self.stage3(x)
        x = self.stage4(out_stage3)
        x = self.stage5(x)
        out_stage6 = self.stage6(x)
        x = self.final_conv(out_stage6)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return out_stage3, out_stage6, x


class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, name):
        super(SelfAttention, self).__init__()
        self.d_out_kq = d_out_kq
        self.W_query = torch.nn.Linear(d_in, d_out_kq)
        self.W_key = torch.nn.Linear(d_in, d_out_kq)
        self.W_value = torch.nn.Linear(d_in, d_out_v)
        self.name = name
        # Normalization layer after attention
        self.norm = torch.nn.LayerNorm(d_out_v)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = torch.matmul(queries, keys.transpose(-1, -2))

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        context_vec = attn_weights.matmul(values)

        output = context_vec
        output = self.norm(output)
        output = self.relu(output)

        return output


class ReverseCrossAttention(torch.nn.Module):
    def __init__(self, d_in_x1, d_in_x2, d_out_kq, d_out_v, reverse):
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = torch.nn.Linear(d_in_x1, d_out_kq)
        self.W_key = torch.nn.Linear(d_in_x2, d_out_kq)
        self.W_value = torch.nn.Linear(d_in_x2, d_out_v)
        self.norm = torch.nn.LayerNorm(d_out_v)
        self.relu = torch.nn.ReLU()
        self.reverse = reverse

    def forward(self, x_1, x_2):
        queries_1 = self.W_query(x_1)
        keys_2 = self.W_key(x_2)
        values_2 = self.W_value(x_2)

        attn_scores = torch.matmul(queries_1, keys_2.transpose(-1, -2))

        attn_weights = torch.softmax(
            attn_scores/self.d_out_kq**0.5, dim=-1
        )

        assert (attn_weights.shape[1] == attn_weights.shape[2])

        if self.reverse:
            # print("RCA!!")
            dimension = attn_weights.shape[1]
            reversed_weights = (1.0-attn_weights)/(dimension-1)
            context_vec = reversed_weights.matmul(values_2)
        else:
            # print("NON RCA")
            context_vec = attn_weights.matmul(values_2)

        output = context_vec
        output = self.norm(output)
        output = self.relu(output)

        return output

def decision(probability):
    return np.random.rand(1)[0] < probability

def eff_net_v2():

    model = efficientnet_v2_m(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        *[model.classifier[i] for i in range(1)])

    extractor = EfficientNetV2MFullFeatureExtractor(model)

    # sys.exit(0)
    return extractor

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
                 reverse,
                 features_only,
                 cross_attention_only):
        super(EffV2MediumAndDistilbertGated, self).__init__()

        self.text_model_name = text_model_name
        self.features_only = features_only
        self.cross_attention_only = cross_attention_only
        
        print("Only features:", self.features_only)
        print("Only cross attention:", self.cross_attention_only)

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

        print("Text model hidden size:",self.text_model.config.hidden_size)
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

        self.num_patches = 16

        hidden_attention_size = 128
        output_attention_size = 96

        cross_attention_hidden_size = 64
        cross_attention_output_size = 48

        input_size_txt = 768
        input_size_img = 1280

        self.txt_patch_size = int(input_size_txt / self.num_patches)
        self.img_patch_size = int(input_size_img / self.num_patches)

        print("txt patch size: ", self.txt_patch_size)
        print("img patch size: ", self.img_patch_size)

        self.self_attention_image = SelfAttention(
            self.img_patch_size, hidden_attention_size, output_attention_size, "Img block")
        self.self_attention_text = SelfAttention(
            self.txt_patch_size, hidden_attention_size, output_attention_size, "Txt block")

        self.cross_attention_1 = ReverseCrossAttention(
            output_attention_size, output_attention_size,
            cross_attention_hidden_size, cross_attention_output_size, reverse)

        self.cross_attention_2 = ReverseCrossAttention(
            output_attention_size, output_attention_size,
            cross_attention_hidden_size, cross_attention_output_size, reverse)

        self.final = torch.nn.Linear(
            cross_attention_output_size*self.num_patches*2, n_classes)

        if self.features_only:
            self.final_features_only_linear = torch.nn.Linear(
            1280+768, n_classes)
        
        if self.cross_attention_only:
            self.cross_attention_only_linear = torch.nn.Linear(
                cross_attention_output_size*self.num_patches*2, n_classes)

        self.final_with_everything = torch.nn.Linear(
            cross_attention_output_size*self.num_patches*2 +
            1280+768, n_classes)
        
        self.final_hierarchical_image = torch.nn.Linear(1280+2560+2048, 512)
        self.final_hierarchical_text = torch.nn.Linear(768*3, 512)
        self.final_hierarchical_all = torch.nn.Linear(512*2, n_classes)

        self.relu = torch.nn.ReLU()
        
        # GRU parameters HIERARCHICAL
        
        modality_dim = 400
        hidden_dim=500
        proj_dim=450
        num_classes=4
        dropout=0.86

        self.modality_dim = modality_dim

        # Bottom row GRUs: Unimodal context modeling
        self.gru_text = torch.nn.GRU(modality_dim, modality_dim, batch_first=True)
        self.gru_audio = torch.nn.GRU(modality_dim, modality_dim, batch_first=True)

        # Bimodal fusion
        self.fusion = Hadamard2(modality_dim)

        # Middle GRU: Context-aware bimodal sequence modeling
        self.gru_bimodal = torch.nn.GRU(modality_dim, hidden_dim, batch_first=True, dropout=0.35)

        # Top layers: classification
        self.dropout1 = torch.nn.Dropout(dropout)
        self.concat_fc = torch.nn.Linear(modality_dim + hidden_dim, proj_dim)
        self.dropout2 = torch.nn.Dropout(dropout)
        
        self.modality_image_to_dim = torch.nn.Linear(1280, modality_dim)
        self.modality_text_to_dim = torch.nn.Linear(768, modality_dim)
        
        self.classifier = torch.nn.Linear(proj_dim, num_classes)        


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


class MM_RCA(EffV2MediumAndDistilbertGated):

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
        _, _, original_image_features = self.image_model(self._images)
        
        # Normalize
        original_text_features = original_text_features / \
            original_text_features.norm(dim=1, keepdim=True)
        original_image_features = original_image_features / \
            original_image_features.norm(dim=1, keepdim=True)

        # Reshape
        bs = original_text_features.shape[0]
        original_text_features_reshaped = \
            torch.reshape(original_text_features,
                          (bs, self.num_patches, self.txt_patch_size))
        original_image_features_reshaped = \
            torch.reshape(original_image_features,
                          (bs, self.num_patches, self.img_patch_size))

        # Self attention
        text_self_attention = self.self_attention_text(
            original_text_features_reshaped)
        img_self_attention = self.self_attention_image(
            original_image_features_reshaped)

        # Cross attention
        complementary_cross_attention_T_I = self.cross_attention_1(
            text_self_attention, img_self_attention)
        complementary_cross_attention_I_T = self.cross_attention_2(
            img_self_attention, text_self_attention)

        # Flatten
        complementary_cross_attention_T_I = torch.flatten(
            complementary_cross_attention_T_I, start_dim=1, end_dim=2)
        complementary_cross_attention_I_T = torch.flatten(
            complementary_cross_attention_I_T, start_dim=1, end_dim=2)

        if self.features_only:
            concat_features = torch.cat(
                (
                    original_image_features,
                    original_text_features
                ), dim=1)

        elif self.cross_attention_only:
            concat_features = torch.cat(
                (
                    complementary_cross_attention_T_I,
                    complementary_cross_attention_I_T
                ), dim=1)

        else:
            # FC layer to output
            concat_features = torch.cat(
            (
            complementary_cross_attention_T_I,
            complementary_cross_attention_I_T,
            original_image_features,
            original_text_features
            ), dim=1)
            

        after_dropout = self.drop(concat_features)

        if self.features_only:
            output = self.final_features_only_linear(after_dropout)
        elif self.cross_attention_only:
            output = self.cross_attention_only_linear(after_dropout)
        else:
            output = self.final_with_everything(after_dropout)

        return output
class Hierarchical(EffV2MediumAndDistilbertGated):

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
            attention_mask=self._attention_mask,
            output_hidden_states=True
        )
        hidden_state = text_output[0]

        # Get the image and text features
        original_text_features = hidden_state[:, 0]
        hidden_states  = text_output.hidden_states
        
       # Extract CLS token embedding (token 0) from layers 2 and 4
        distilbert_layer_2 = hidden_states[2][:, 0, :]  # Layer 2
        distilbert_layer_4 = hidden_states[4][:, 0, :]  # Layer 4
        out_stage_3, out_stage_6, original_image_features = self.image_model(self._images)
        
        out_stage_3_avg_pooling = \
            torch.nn.AvgPool2d(kernel_size=7, stride=7)(out_stage_3)
        # → becomes [bs, 160, 4, 4] 
            
        out_stage_6_avg_pooling = \
            torch.nn.AvgPool2d(kernel_size=6, stride=6)(out_stage_6)
        # → becomes [bs, 512, 2, 2] 

        out_stage_3_avg_pooling = \
            out_stage_3_avg_pooling.view(out_stage_3_avg_pooling.size(0), -1)
            
        out_stage_6_avg_pooling = \
            out_stage_6_avg_pooling.view(out_stage_6_avg_pooling.size(0), -1)         

        out_stage_3_flattened = out_stage_3_avg_pooling.flatten(start_dim=1)
        out_stage_6_flattened = out_stage_6_avg_pooling.flatten(start_dim=1) 

        out_stage_3_flattened = out_stage_3_flattened / \
            out_stage_3_flattened.norm(dim=1, keepdim=True)
        out_stage_6_flattened = out_stage_6_flattened / \
            out_stage_6_flattened.norm(dim=1, keepdim=True)
        original_image_features = original_image_features / \
            original_image_features.norm(dim=1, keepdim=True)

        distilbert_layer_2 = distilbert_layer_2 / \
            distilbert_layer_2.norm(dim=1, keepdim=True)
        distilbert_layer_4 = distilbert_layer_4 / \
            distilbert_layer_4.norm(dim=1, keepdim=True)
        original_text_features = original_text_features / \
            original_text_features.norm(dim=1, keepdim=True)

        concat_features_image = torch.cat(
            (
                original_image_features,
                out_stage_3_flattened,
                out_stage_6_flattened,
            ), dim=1)
        
        concat_features_text = torch.cat(
            (
                original_text_features,
                distilbert_layer_2,
                distilbert_layer_4,
            ), dim=1)        

        after_dropout_image = self.drop(concat_features_image)
        after_dropout_text = self.drop(concat_features_text)
        
        image = self.final_hierarchical_image(after_dropout_image)
        text = self.final_hierarchical_text(after_dropout_text)

        image = self.relu(image)
        text = self.relu(text)

        output = self.final_hierarchical_all(
            torch.cat((image, text), dim=1)
        )

        return output    



class Hadamard2(torch.nn.Module):
    
    def __init__(self, dim):
        super(Hadamard2, self).__init__()
        self.kernel1 = torch.nn.Parameter(torch.randn(dim))
        self.kernel2 = torch.nn.Parameter(torch.randn(dim))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x1, x2):
        return torch.tanh(x1 * self.kernel1 + x2 * self.kernel2 + self.bias)



class HierarchicalBimodalFusion(EffV2MediumAndDistilbertGated):
    
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
            attention_mask=self._attention_mask,
            output_hidden_states=True
        )
        hidden_state = text_output[0]

        # Get the image and text features
        original_text_features = hidden_state[:, 0]
              
        _ ,_ , original_image_features = self.image_model(self._images)
        
        original_image_features = original_image_features / \
            original_image_features.norm(dim=1, keepdim=True)
            
        original_text_features = original_text_features / \
            original_text_features.norm(dim=1, keepdim=True)            
        
        # Input: (B, T, 2 × modality_dim)
        x_text = self.modality_text_to_dim(original_text_features)
        x_image = self.modality_image_to_dim(original_image_features)

        # Unimodal context GRUs
        ctx_text, _ = self.gru_text(x_text)     # (B, T, modality_dim)
        ctx_audio, _ = self.gru_audio(x_image)  # (B, T, modality_dim)

        # Bimodal fusion
        fused = self.fusion(ctx_text, ctx_audio)  # (B, T, modality_dim)

        # Bimodal context GRU
        ctx_fused, _ = self.gru_bimodal(fused)    # (B, T, hidden_dim)
        ctx_fused = self.dropout1(ctx_fused)

        # Fusion + classification
        combined = torch.cat([fused, ctx_fused], dim=-1)  # (B, T, modality_dim + hidden_dim)
        proj = self.dropout2(self.relu(self.concat_fc(combined)))
        logits = self.classifier(proj)  # (B, T, num_classes)

        return logits
