import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm
import random

from utilities.constants import *
from utilities.device import get_device

from .positional_encoding import PositionalEncoding
from .rpr import TransformerEncoderRPR, TransformerEncoderLayerRPR
import json
# MusicTransformer
class MusicTransformer(nn.Module):
    def __init__(self, n_layers=6, num_heads=8, d_model=512, dim_feedforward=1024,
                 dropout=0.1, max_sequence_midi=2048, max_sequence_chord=300,  rpr=False):
        super(MusicTransformer, self).__init__()

        self.dummy      = DummyDecoder()
        self.nlayers    = n_layers
        self.nhead      = num_heads
        self.d_model    = d_model
        self.d_ff       = dim_feedforward
        self.dropout    = dropout
        self.max_seq_midi    = max_sequence_midi
        self.max_seq_chord    = max_sequence_chord
        self.rpr        = rpr

        # Input embedding for video and music features
        self.embedding = nn.Embedding(CHORD_SIZE, self.d_model)

        # self.embedding_key = nn.Embedding(1, self.d_model)
        self.embedding_root = nn.Embedding(CHORD_ROOT_SIZE, self.d_model)
        self.embedding_attr = nn.Embedding(CHORD_ATTR_SIZE, self.d_model)

        self.positional_encoding = PositionalEncoding(self.d_model, self.dropout, self.max_seq_chord)
        self.Linear_chord     = nn.Linear(self.d_model+1, self.d_model)

        # Base transformer
        if(not self.rpr):
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy
            )
        # RPR Transformer
        else:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(self.d_model, self.nhead, self.d_ff, self.dropout, er_len=self.max_seq_chord)

            encoder = TransformerEncoderRPR(encoder_layer, self.nlayers, encoder_norm)
            self.transformer = nn.Transformer(
                d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.nlayers,
                num_decoder_layers=0, dropout=self.dropout, # activation=self.ff_activ,
                dim_feedforward=self.d_ff, custom_decoder=self.dummy, custom_encoder=encoder
            )
        # Final output is a softmaxed linear layer
        self.Wout       = nn.Linear(self.d_model, CHORD_SIZE)
        self.Wout_root       = nn.Linear(self.d_model, CHORD_ROOT_SIZE)
        self.Wout_attr       = nn.Linear(self.d_model, CHORD_ATTR_SIZE)
        self.softmax    = nn.Softmax(dim=-1)

    # forward
    def forward(self, x, x_root, x_attr, feature_key, mask=True):
        if(mask is True):
            mask = self.transformer.generate_square_subsequent_mask(x.shape[1]).to(get_device())
        else:
            mask = None

        ### Chord + Key (DECODER) ###
        # x = self.embedding(x)
        
        x_root = self.embedding_root(x_root)
        x_attr = self.embedding_attr(x_attr)
        x = x_root + x_attr

        feature_key_padded = torch.full((x.shape[0], x.shape[1], 1), feature_key.item())
        feature_key_padded = feature_key_padded.to(get_device())
        x = torch.cat([x, feature_key_padded], dim=-1)
        xf = self.Linear_chord(x)

        ### POSITIONAL ENCODING ###
        xf = xf.permute(1,0,2) # -> (max_seq-1, batch_size, d_model)
        xf = self.positional_encoding(xf)
        
        ### TRANSFORMER ###
        x_out = self.transformer(src=xf, tgt=xf, tgt_mask=mask)
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

    # generate
    def generate(self, feature_key=None, primer=None, primer_root=None, primer_attr=None, target_seq_length=300, beam=0, beam_chance=1.0):
        assert (not self.training), "Cannot generate while in training mode"

        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        print("Generating sequence of max length:", target_seq_length)
        gen_seq = torch.full((1,target_seq_length), CHORD_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_root = torch.full((1,target_seq_length), CHORD_ROOT_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_seq_attr = torch.full((1,target_seq_length), CHORD_ATTR_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        
        num_primer = len(primer)

        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_seq_root[..., :num_primer] = primer_root.type(TORCH_LABEL_TYPE).to(get_device())
        
        gen_seq_attr[..., :num_primer] = primer_attr.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            # gen_seq_batch     = gen_seq.clone()
            # y = self.softmax(self.forward(gen_seq[..., :cur_i]))[..., :CHORD_END]
            y = self.softmax( self.forward( gen_seq[..., :cur_i], gen_seq_root[..., :cur_i], gen_seq_attr[..., :cur_i], feature_key) )[..., :CHORD_END]
            
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
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                #print("next token:",next_token)
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

class DummyDecoder(nn.Module):
    def __init__(self):
        super(DummyDecoder, self).__init__()
    def forward(self, tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask,memory_key_padding_mask):
        return memory
