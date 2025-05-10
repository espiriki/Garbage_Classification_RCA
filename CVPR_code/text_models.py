import torch
import torch.nn as nn
from transformers import BertModel, DistilBertModel, RobertaModel, BartForSequenceClassification, MobileBertForSequenceClassification
from transformers import BertConfig, DistilBertConfig, RobertaConfig, BartConfig
from torchvision.models import *
from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer, BartTokenizer, MobileBertTokenizer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config, MobileBertConfig

class DistilBert(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(DistilBert, self).__init__()
        self.name = "distilbert-base-uncased"
        self.model = DistilBertModel.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/distilbert/modeling_distilbert.py#L729
    def forward(self, _input_ids, _attention_mask):
        distilbert_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )
        # dim = HIDDEN_SIZE
        # bs = BATCH_SIZE
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)

    def get_tokenizer(self):
        return DistilBertTokenizer.from_pretrained(self.name)

    def get_max_token_size(self):
        return DistilBertConfig().max_position_embeddings


class Roberta(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(Roberta, self).__init__()
        self.name = "roberta-base"
        self.model = RobertaModel.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, _input_ids, _attention_mask):
        roberta_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )

        hidden_state = roberta_output[0]
        pooled_output = hidden_state[:, 0]
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)

    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.name)

    def get_max_token_size(self):
        return RobertaConfig().max_position_embeddings

class Bert(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(Bert, self).__init__()
        self.name = "bert-base-uncased"
        self.model = BertModel.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(p=drop_ratio)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, _input_ids, _attention_mask):
        bert_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )

        hidden_state = bert_output[0]
        pooled_output = hidden_state[:, 0]
        drop_output = self.drop(pooled_output)

        return self.out(drop_output)
    
    def get_tokenizer(self):
        return BertTokenizer.from_pretrained(self.name)

    def get_max_token_size(self):
        return BertConfig().max_position_embeddings
    
class Bart(nn.Module):
    def __init__(self, n_classes, drop_ratio):
        super(Bart, self).__init__()
        self.name = "facebook/bart-large"
        self.model = BartForSequenceClassification.from_pretrained(self.name)

        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classification_head.dropout = nn.Dropout(p=drop_ratio)
        self.model.classification_head.out_proj = \
            nn.Linear(self.model.classification_head.dense.out_features, n_classes)

    def forward(self, _input_ids, _attention_mask):
        bart_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )

        return bart_output[0]
    
    def get_tokenizer(self):
        return BartTokenizer.from_pretrained(self.name)

    def get_max_token_size(self):
        return BartConfig().max_position_embeddings

class GPT2(nn.Module):
    def __init__(self, n_classes):
        super(GPT2, self).__init__()
        self.name = "gpt2"
        self.model = GPT2ForSequenceClassification.from_pretrained(self.name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.score = \
            nn.Linear(self.model.score.in_features, n_classes)

    def forward(self, _input_ids, _attention_mask):
        gpt2_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )

        return gpt2_output[0]

    def get_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(self.name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_max_token_size(self):
        return GPT2Config().n_positions
    
class MobileBERT(nn.Module):
    def __init__(self, n_classes):
        super(MobileBERT, self).__init__()
        self.name = "google/mobilebert-uncased"
        self.model = MobileBertForSequenceClassification.from_pretrained(self.name)
    
        # Freeze all layers for TL
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = \
            nn.Linear(self.model.classifier.in_features, n_classes)          

    def forward(self, _input_ids, _attention_mask):
        mobile_bert_output = self.model(
            input_ids=_input_ids,
            attention_mask=_attention_mask
        )

        return mobile_bert_output[0]

    def get_tokenizer(self):
        tokenizer = MobileBertTokenizer.from_pretrained(self.name)
        return tokenizer

    def get_max_token_size(self):
        return MobileBertConfig().max_position_embeddings