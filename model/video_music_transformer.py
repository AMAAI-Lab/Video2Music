import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random
import numpy as np
from utilities.constants import *
from utilities.device import get_device
from .positional_encoding import PositionalEncoding
from .rpr import TransformerDecoderRPR, TransformerDecoderLayerRPR
from datetime import datetime
import json


class VideoMusicTransformer(nn.Module):
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi =2048, max_sequence_video=300, max_sequence_chord=300, total_vf_dim = 0, rpr=False):
        super(VideoMusicTransformer, self).__init__()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_video    = max_sequence_video
        self.max_seq_chord    = max_sequence_chord
        self.rpr        = rpr

        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)
        
        self.total_vf_dim = total_vf_dim
        self.Linear_vis     = nn.Linear(self.total_vf_dim, self.d_model)
        self.Linear_chord     = nn.Linear(self.d_model+1, self.d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq_chord)
        self.positional_encoding_video = PositionalEncoding(self.d_model, self.dropout, self.max_seq_video)

        # Add condition (minor or major)
        self.condition_linear = nn.Linear(1, self.d_model)
        
        # Base transformer
        if(not self.rpr):
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff
            )
        # RPR Transformer
        else:
            decoder_norm = LayerNorm(self.d_model)
            decoder_layer = TransformerDecoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq_chord)
            decoder = TransformerDecoderRPR(decoder_layer, self.nlayers, decoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=self.nlayers, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=decoder
            )        
        
        self.Wout       = nn.Linear(self.d_model, CHORD_SIZE)
        self.Wout_root       = nn.Linear(self.d_model, CHORD_ROOT_SIZE)
        self.Wout_attr       = nn.Linear(self.d_model, CHORD_ATTR_SIZE)
        self.softmax    = nn.Softmax(dim=-1)
    
    def forward(self, x, x_root, x_attr, feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None
        
        x_root = self.embedding_root(x_root)
        x_attr = self.embedding_attr(x_attr)
        x = x_root + x_attr

        feature_key_padded = torch.full((x.shape[0], x.shape[1], 1), feature_key.item())
        feature_key_padded = feature_key_padded.to(get_device())
        x = torch.cat([x, feature_key_padded], dim=-1)

        xf = self.Linear_chord(x)

        ### Video (SemanticList + SceneOffset + Motion + Emotion) (ENCODER) ###
        vf_concat = feature_semantic_list[0].float()

        for i in range(1, len(feature_semantic_list)):
            vf_concat = torch.cat( (vf_concat, feature_semantic_list[i].float()), dim=2)            
        
        vf_concat = torch.cat([vf_concat, feature_scene_offset.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        vf_concat = torch.cat([vf_concat, feature_motion.unsqueeze(-1).float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        vf_concat = torch.cat([vf_concat, feature_emotion.float()], dim=-1) # -> (max_seq_video, batch_size, d_model+1)
        vf = self.Linear_vis(vf_concat)
        
        ### POSITIONAL ENCODING ###
        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        vf = vf.permute(1,0,2) # -> (max_seq_video, batch_size, d_model)
        xf = self.positional_encoding(xf)
        vf = self.positional_encoding_video(vf)

        ### TRANSFORMER ###
        x_out = self.transformer(src=vf, tgt=xf, tgt_mask=mask)
        x_out = x_out.permute(1,0,2)

        if IS_SEPERATED:
            y_root = self.Wout_root(x_out)
            y_attr = self.Wout_attr(x_out)
            del mask
            return y_root, y_attr
        else:
            y = self.Wout(x_out)
            del mask
            return y
    
    def generate(self, feature_semantic_list = [], feature_key=None, feature_scene_offset=None, feature_motion=None, feature_emotion=None,
                 primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0, max_conseq_N = 0, max_conseq_chord = 2):
        
        assert (not self.training), "Cannot generate while in training mode"
        print("Generating sequence of max length:", target_seq_length)

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], 
                                           feature_semantic_list, feature_key, feature_scene_offset, feature_motion, feature_emotion) )[..., :CHORD_END]
            
            token_probs = y[:, cur_i-1, :]
            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)
            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)
                beam_rows = top_i // CHORD_SIZE
                beam_cols = top_i % CHORD_SIZE
                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols
            else:
                # token_probs.shape : [1, 157] 
                # 0: N, 1: C, ... , 156: B:maj7
                # 157 chordEnd 158 padding
                if max_conseq_N == 0:
                    token_probs[0][0] = 0.0
                isMaxChord = True
                if cur_i >= max_conseq_chord :
                    preChord = gen_seq[0][cur_i-1].item()      
                    for k in range (1, max_conseq_chord):
                        if preChord != gen_seq[0][cur_i-1-k].item():
                            isMaxChord = False
                else:
                    isMaxChord = False
                
                if isMaxChord:
                    preChord = gen_seq[0][cur_i-1].item()
                    token_probs[0][preChord] = 0.0
                
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token
                gen_chord = chordInvDic[ str( next_token.item() ) ]
                
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = 1
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                elif len(chord_arr) == 2:
                    chordRootID = chordRootDic[chord_arr[0]]
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    chordRootID = torch.tensor([chordRootID]).to(get_device())
                    chordAttrID = torch.tensor([chordAttrID]).to(get_device())
                    gen_seq_root[:, cur_i] = chordRootID
                    gen_seq_attr[:, cur_i] = chordAttrID
                    
                # Let the transformer decide to end if it wants to
                if(next_token == CHORD_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break
            cur_i += 1
            if(cur_i % 50 == 0):
                print(cur_i, "/", target_seq_length)
        return gen_seq[:, :cur_i]

