import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import difflib
from typing import Optional, Tuple, Union

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, BertTokenizer, BertModel, Wav2Vec2Model, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
from .motion_encoder import VQEncoderV6


def audio_to_time_aligned_text_features(inputs, processor, model, tokenizer, bert_model):  
    with torch.no_grad():
        logits = model(inputs.input_values).logits  # shape: (1, time_steps, vocab_size)

    predicted_ids_per_timestep = torch.argmax(logits, dim=-1)  # shape: (1, time_steps)
    predicted_ids_per_timestep = predicted_ids_per_timestep[0].cpu().numpy()
    vocab = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    tokens_per_timestep = [id_to_token[id] for id in predicted_ids_per_timestep]

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    inputs_bert = tokenizer(transcription, return_tensors='pt')
    input_ids = inputs_bert['input_ids'][0]  
    tokens_bert = tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        outputs_bert = bert_model(**inputs_bert.to(inputs.input_values.device))
    all_token_embeddings = outputs_bert.last_hidden_state[0]  
    per_timestep_chars = []
    per_timestep_char_indices = []
    for idx, t in enumerate(tokens_per_timestep):
        if t not in ('<pad>', '|'):
            per_timestep_chars.append(t.lower())
            per_timestep_char_indices.append(idx)
    bert_chars = []
    bert_char_indices = []
    for idx, token in enumerate(tokens_bert):
        if token in ('[CLS]', '[SEP]'):
            continue
        token_str = token.replace('##', '')
        for c in token_str:
            bert_chars.append(c)
            bert_char_indices.append(idx)

    s = difflib.SequenceMatcher(None, per_timestep_chars, bert_chars)
    opcodes = s.get_opcodes()
    per_timestep_to_bert_token_idx = {}
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(i2 - i1):
                per_timestep_idx = per_timestep_char_indices[i1 + k]
                bert_token_idx = bert_char_indices[j1 + k]
                per_timestep_to_bert_token_idx[per_timestep_idx] = bert_token_idx
    features_per_timestep = []
    check = []
    for i, per_token in enumerate(tokens_per_timestep):
        if i == 0:
            embedding = all_token_embeddings[0]
            check.append("cls")
        elif per_token in ('<pad>', '|'):
            embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
            check.append(0)
        else:
            if i in per_timestep_to_bert_token_idx:
                bert_idx = per_timestep_to_bert_token_idx[i]
                embedding = all_token_embeddings[bert_idx]
                check.append(tokens_bert[bert_idx])
            else:
                embedding = torch.zeros(all_token_embeddings.shape[-1]).to(inputs.input_values.device)
                check.append(0)
        features_per_timestep.append(embedding)
    features_per_timestep = torch.stack(features_per_timestep)  

    updated_check = check.copy()
    for i in range(len(check)):
        if check[i] == 0:
            left = i - 1
            right = i + 1
            left_found = False
            right_found = False

            while left >= 0:
                if check[left] != 0:
                    left_found = True
                    break
                left -= 1

            while right < len(check):
                if check[right] != 0:
                    right_found = True
                    break
                right += 1

            if left_found and right_found:
                if (i - left) <= (right - i):
                    nearest = left
                else:
                    nearest = right
            elif left_found:
                nearest = left
            elif right_found:
                nearest = right
            else:
                continue
            updated_check[i] = updated_check[nearest]
            features_per_timestep[i] = features_per_timestep[nearest]
    features_per_timestep = features_per_timestep.unsqueeze(0)
    return transcription, features_per_timestep, all_token_embeddings 


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=20, max_seq_len=64): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, seq_len, embed_dim = query.size()

        # Linear projections
        Q = self.query_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply final linear projection
        output = self.out_proj(attn_output)
        return output, attn_weights  # Return the per-head attention weights


def reinitialize_weights(module):
    for submodule in module.modules():
        weight = getattr(submodule, 'weight', None)
        if weight is not None and isinstance(weight, torch.Tensor) and weight.dim() >= 2:
            torch.nn.init.xavier_uniform_(weight)
            print("init")
        elif weight is not None and isinstance(weight, torch.Tensor):
            torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            print("init")
        bias = getattr(submodule, 'bias', None)
        if bias is not None and isinstance(bias, torch.Tensor):
            torch.nn.init.zeros_(bias)
        

class WrapedMotionCNN(nn.Module):
    def __init__(self, args):
        super(WrapedMotionCNN, self).__init__()
        self.args = args
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.args.motion_f,  # This should match the hidden size of the Wav2Vec2 model
            nhead=8,      # Number of attention heads
            dim_feedforward=self.args.hidden_size,  # The feedforward network dimension
            dropout=0.1,   # Dropout rate
            batch_first=True
        )
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = self.args.motion_f
        args_top.vae_test_dim = self.args.motion_dim
        self.feature_extractor = VQEncoderV6(args_top) 

     
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 6
        args_top.vae_length = self.args.motion_f
        args_top.vae_test_dim = self.args.motion_dim + self.args.motion_f
      
        self.encoder_cnn = VQEncoderV6(args_top) 
        self.pos_encoding = PeriodicPositionalEncoding(d_model=self.args.motion_f, period=20, max_seq_len=64, dropout=0.0)
        self.encoder_trans = nn.TransformerEncoder(encoder_layer, num_layers=1) # Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').encoder

    def forward(self, 
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ):
        low_level = self.feature_extractor(inputs)
        # print(low_level.shape, inputs.shape)
        hidden_states = self.encoder_cnn(torch.cat([low_level.detach(), inputs], dim=-1))
        hidden_states = self.pos_encoding(hidden_states)
        hidden_states = self.encoder_trans(hidden_states)
        return {
            "low_level": low_level,
            "high_level": hidden_states
        }
        

class WrapedWav2Vec(nn.Module):
    def __init__(self):
        super(WrapedWav2Vec, self).__init__()
        self.feature_extractor = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').feature_extractor
        self.feature_projection = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').feature_projection
        self.encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').encoder
        # print(self.encoder)
        self.encoder.layers = self.encoder.layers[:1]
        # print(self.encoder)
        self.proj_down = nn.Linear(768,512)
        # print(bug)
    
    def forward(self, 
        inputs,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ):
        finetune_audio_low = self.feature_extractor(inputs).transpose(1, 2)
        hidden_states, _ = self.feature_projection(finetune_audio_low.detach())
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        hidden_states = self.proj_down(hidden_states)
        # print(hidden_states.shape)
        return {
            "low_level": finetune_audio_low,
            "high_level": hidden_states
        }


class JointEmbedding(nn.Module):
    def __init__(self, args):
        super(JointEmbedding, self).__init__()
        self.args = args.model   
        self.audio_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        self.config_wav2vec = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base-960h')
        # self.audio_encoder_fintune = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').feature_extractor
        self.audio_encoder_fintune = WrapedWav2Vec()
        # print(self.audio_encoder_fintune)
        # print(bug)
        
        self.asr = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.audio_low_mapping = MLP(512+512, self.args.hidden_size, self.args.audio_f)
        self.audio_high_mapping = MLP(512+512+512, self.args.hidden_size, self.args.audio_f)
        # self.audio_down_proj_1 = nn.Linear(768, 512)
        self.audio_down_proj_2 = nn.Linear(768, 512)
        self.audio_down_proj_3 = nn.Linear(768, 512)
        # self.audio_sa = nn.MultiheadAttention(embed_dim=self.args.audio_f, num_heads=8, batch_first=True)
        self.audio_sa = CustomMultiheadAttention(embed_dim=self.args.audio_f, num_heads=8,)

        self.motion_encoder_fintune = WrapedMotionCNN(self.args)
        self.motion_low_mapping = MLP(self.args.motion_f, self.args.hidden_size, self.args.motion_f)
        self.motion_high_mapping = MLP(self.args.motion_f, self.args.hidden_size, self.args.motion_f)
        # self.motion_sa = nn.MultiheadAttention(embed_dim=self.args.audio_f, num_heads=8, batch_first=True)
        self.motion_sa = CustomMultiheadAttention(embed_dim=self.args.audio_f, num_heads=8,)
        
        self.down_sample = 2 # for downsample 30 fps motion to 15
        self.smplx_model = None
        self.get_motion_reps = None
        self.audio_to_time_aligned_text_features = audio_to_time_aligned_text_features
        self.low_temp = nn.Parameter(torch.tensor(0.07))
        self.low_level_loss_fn = None
        self.high_temp = nn.Parameter(torch.tensor(0.07))
        self.high_level_loss_fn = None

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)
    
    def forward(self, in_audio=None, in_motion=None, cached_audio_low=None, cached_audio_high=None, cached_rep15d=None):
        # motion feature
        if cached_rep15d is not None:
            in_motion = cached_rep15d[:,::self.down_sample]
        else:
            in_motion = self.get_motion_reps(in_motion, self.smplx_model)["rep15d"][:,::self.down_sample]
        
        motion_features = self.motion_encoder_fintune(in_motion)
        raw_motion_low = motion_features["low_level"] # self.motion_encoder_low(in_motion)
        raw_motion_high = motion_features["high_level"] # self.motion_encoder_high(torch.cat([raw_motion_low.detach(), in_motion], dim=-1))

        motion_low = self.motion_low_mapping(raw_motion_low)
        motion_high = self.motion_high_mapping(raw_motion_high)
        motion_high_att, motion_high_weight = self.motion_sa(motion_high, motion_high, motion_high)
        bs, n, c = motion_high.shape
        # print("a:", motion_high_weight[:, :, 0, :].unsqueeze(2).shape, "b:", motion_high.transpose(1, 2).view(bs, 8, c//8, n).shape)
        motion_high_att_before_sum = motion_high_weight[:, :, 0, :].unsqueeze(2) * motion_high.transpose(1, 2).view(bs, 8, c//8, n)
        motion_high_att_before_sum = motion_high_att_before_sum.reshape(bs, c, n).transpose(1, 2)
        motion_low = F.interpolate(motion_low.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        motion_high_att = F.interpolate(motion_high_att.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        motion_high_att_before_sum = F.interpolate(motion_high_att_before_sum.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        motion_cls = motion_high_att[:, 0]

        # audio feature
        if cached_audio_low is not None:
            raw_audio_low = cached_audio_low
            raw_audio_high = torch.cat([self.audio_down_proj_2(cached_audio_high[:, :, :768]), self.audio_down_proj_3(cached_audio_high[:, :, 768:])], dim=-1)
            
            audio_list = [i.cpu().numpy() for i in in_audio]
            inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(in_audio.device)
            finetune_audio = self.audio_encoder_fintune(inputs.input_values)
            finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
            diff = raw_audio_low.shape[1] - finetune_audio_low.shape[1]
            if diff > 0:
                finetune_audio_low = torch.cat([finetune_audio_low, finetune_audio_low[:, -diff:]], dim=1)
            diff = raw_audio_high.shape[1] - finetune_audio_high.shape[1]
            if diff > 0:
                finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)
            raw_audio_low = torch.cat([raw_audio_low, finetune_audio_low], dim=-1) # bs, t, 1024
        else:
            print("error! must have cached audio in training")
        
        # print(raw_audio_low.shape, raw_audio_high.shape, "before")

        raw_audio_low = F.interpolate(raw_audio_low.transpose(1, 2), scale_factor=30/50, mode='linear', align_corners=True).transpose(1, 2) 
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), scale_factor=15/50, mode='linear', align_corners=True).transpose(1, 2)
        finetune_audio_high = F.interpolate(finetune_audio_high.transpose(1, 2), scale_factor=15/50, mode='linear', align_corners=True).transpose(1, 2)  
        # print(raw_audio_low.shape, raw_audio_high.shape, "after")
        audio_low = self.audio_low_mapping(raw_audio_low)
        raw_audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        # print(finetune_audio_high.shape, raw_audio_high.shape)
        audio_high = self.audio_high_mapping(raw_audio_high)
        audio_high_att, audio_high_weight = self.audio_sa(audio_high, audio_high, audio_high)
        bs, n, c = audio_high.shape
        audio_high_att_before_sum = audio_high_weight[:, :, 0, :].unsqueeze(2) * audio_high.transpose(1, 2).view(bs, 8, c//8, n)
        audio_high_att_before_sum = audio_high_att_before_sum.reshape(bs, c, n).transpose(1, 2)
        audio_high_att = F.interpolate(audio_high_att.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        audio_high_att_before_sum = F.interpolate(audio_high_att_before_sum.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        audio_cls = audio_high_att[:, 0]
        # low_infonce, low_acc = self.low_level_loss_fn(audio_low, motion_low, learned_temp=self.low_temp)
        
        # fix temp to 0.1 is better than learned temp
        low_infonce, low_acc = self.low_level_loss_fn(audio_low, motion_low)
        high_infonce = self.high_level_loss_fn(audio_cls, motion_cls)
        return {
            "audio_low":audio_low,
            "audio_high":audio_high_att,
            "audio_cls":audio_cls,
            "audio_high_weight":audio_high_att_before_sum,
            "motion_low":motion_low,
            "motion_high":motion_high_att,
            "motion_cls":motion_cls,
            "motion_high_weight":motion_high_att_before_sum,
            "low_level_loss": [low_infonce, low_acc],
            "high_level_loss": high_infonce
            }

    def get_audio_features(self, in_audio):
        audio_list = [i.cpu().numpy() for i in in_audio]
        inputs = self.audio_processor(audio_list, sampling_rate=16000, return_tensors="pt", padding=True).to(in_audio.device)
        raw_audio_low = self.audio_encoder.feature_extractor(inputs.input_values).transpose(1, 2)
        raw_audio_low = raw_audio_low
            
        finetune_audio = self.audio_encoder_fintune(inputs.input_values)
        finetune_audio_low, finetune_audio_high = finetune_audio["low_level"], finetune_audio["high_level"]
        diff = raw_audio_low.shape[1] - finetune_audio_low.shape[1]
        if diff > 0:
            finetune_audio_low = torch.cat([finetune_audio_low, finetune_audio_low[:, -diff:]], dim=1)
        raw_audio_low = torch.cat([raw_audio_low, finetune_audio_low], dim=-1)

        raw_audio_high = self.audio_encoder(inputs.input_values).last_hidden_state
        
        diff = raw_audio_high.shape[1] - finetune_audio_high.shape[1]
        if diff > 0:
            finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)
        # print(raw_audio_high.shape, finetune_audio_high.shape)

        _, bert_time_aligned_text, _ = audio_to_time_aligned_text_features(inputs, self.audio_processor, self.asr, self.bert_tokenizer, self.bert_model)
        raw_audio_high = torch.cat([raw_audio_high, bert_time_aligned_text], dim=2)
        raw_audio_high = torch.cat([self.audio_down_proj_2(raw_audio_high[:, :, :768]), self.audio_down_proj_3(raw_audio_high[:, :, 768:])], dim=-1)

        raw_audio_low = F.interpolate(raw_audio_low.transpose(1, 2), scale_factor=30/50, mode='linear', align_corners=True).transpose(1, 2) 
        raw_audio_high = F.interpolate(raw_audio_high.transpose(1, 2), scale_factor=15/50, mode='linear', align_corners=True).transpose(1, 2)
        finetune_audio_high = F.interpolate(finetune_audio_high.transpose(1, 2), scale_factor=15/50, mode='linear', align_corners=True).transpose(1, 2) 
        
        if raw_audio_low.shape[1] % 2 == 1:
            raw_audio_low = torch.cat([raw_audio_low, raw_audio_low[:, -1:]], dim=1)
        diff = raw_audio_low[:, ::2].shape[1] - raw_audio_high.shape[1]
        if diff > 0:
            raw_audio_high = torch.cat([raw_audio_high, raw_audio_high[:, -diff:]], dim=1)
            finetune_audio_high = torch.cat([finetune_audio_high, finetune_audio_high[:, -diff:]], dim=1)

        audio_low = self.audio_low_mapping(raw_audio_low)
        # print(audio_low.shape[1]//2, raw_audio_high.shape[1])
        raw_audio_high = torch.cat([finetune_audio_high, raw_audio_high], dim=-1)
        audio_high = self.audio_high_mapping(raw_audio_high)
        audio_high_att, audio_high_weight = self.audio_sa(audio_high, audio_high, audio_high)
        bs, n, c = audio_high.shape
        audio_high_att_before_sum = audio_high_weight[:, :, 0, :].unsqueeze(2) * audio_high.transpose(1, 2).view(bs, 8, c//8, n)
        audio_high_att_before_sum = audio_high_att_before_sum.reshape(bs, c, n).transpose(1, 2)
        audio_high_att = F.interpolate(audio_high_att.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        audio_high_att_before_sum = F.interpolate(audio_high_att_before_sum.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        audio_cls = audio_high_att[:, 0]
        return {
            "audio_low":audio_low,
            "audio_high":audio_high_att,
            "audio_cls":audio_cls,
            "audio_high_weight":audio_high_att_before_sum,
            }

    def get_motion_features(self, in_motion):
        original_length = in_motion.shape[1]
         
        in_motion = self.get_motion_reps(in_motion, self.smplx_model)["rep15d"][:,::self.down_sample]
        motion_features = self.motion_encoder_fintune(in_motion)
        raw_motion_low = motion_features["low_level"] # self.motion_encoder_low(in_motion)
        raw_motion_high = motion_features["high_level"] # self.motion_encoder_high(torch.cat([raw_motion_low.detach(), in_motion], dim=-1))
        motion_low = self.motion_low_mapping(raw_motion_low)
        motion_high = self.motion_high_mapping(raw_motion_high)
        
        motion_high_att, motion_high_weight = self.motion_sa(motion_high, motion_high, motion_high)
        bs, n, c = motion_high.shape
        motion_high_att_before_sum = motion_high_weight[:, :, 0, :].unsqueeze(2) * motion_high.transpose(1, 2).view(bs, 8, c//8, n)
        motion_high_att_before_sum = motion_high_att_before_sum.reshape(bs, c, n).transpose(1, 2)
        motion_low = F.interpolate(motion_low.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        motion_high_att = F.interpolate(motion_high_att.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        motion_high_att_before_sum = F.interpolate(motion_high_att_before_sum.transpose(1, 2), scale_factor=2, mode='linear', align_corners=True).transpose(1, 2)
        
        # if motion_low.shape[1] - 
        motion_low = motion_low[:, :original_length]
        motion_high_att = motion_high_att[:, :original_length]
        motion_high_att_before_sum = motion_high_att_before_sum[:, :original_length]

        motion_cls = motion_high_att[:, 0]
        # print(original_length, motion_low.shape, motion_high_att.shape, motion_high_att_before_sum.shape)
        return {
            "motion_low":motion_low,
            "motion_high":motion_high_att,
            "motion_cls":motion_cls,
            "motion_high_weight":motion_high_att_before_sum,
            }
  