import torch
from thirdparty.midiprocessor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_VEL, RANGE_TIME_SHIFT

#Proposed (AMT l0.4)
# VERSION = "v27_video_rpr_nosep_l0.4"
VERSION = "AMT"

#Best Baseline (MT)
# VERSION = "v27_novideo_rpr_nosep"

IS_SEPERATED = False # True : seperated chord quality and root output
RPR = True
IS_VIDEO = True

GEN_MODEL = "Video Music Transformer"
# LSTM
# Transformer
# Music Transformer
# Video Music Transformer

LOSS_LAMBDA = 0.4 # lamda * chord  +  ( 1-lamda ) * emotion

EMOTION_THRESHOLD = 0.80

VIS_MODELS = "2d/clip_l14p"
SPLIT_VER = "v1"

MUSIC_TYPE = "lab_v2_norm"
# - midi_prep
# - lab
# - lab_v2
# - lab_v2_norm
# ----------------------------------------- #

VIS_ABBR_DIC = {
    "2d/clip_l14p" : "clip_l14p", # NEW
}

vis_arr = VIS_MODELS.split(" ")
vis_arr.sort()
vis_abbr_path = ""
for v in vis_arr:
    vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
vis_abbr_path = vis_abbr_path[1:]

VIS_MODELS_PATH = vis_abbr_path
VIS_MODELS_SORTED = " ".join(vis_arr)

# CHORD
CHORD_END               = 157
CHORD_PAD               = CHORD_END + 1 
CHORD_SIZE              = CHORD_PAD + 1

# CHORD_ROOT
CHORD_ROOT_END               = 13
CHORD_ROOT_PAD               = CHORD_ROOT_END + 1
CHORD_ROOT_SIZE              = CHORD_ROOT_PAD + 1

# CHORD_ATTR
CHORD_ATTR_END               = 14
CHORD_ATTR_PAD               = CHORD_ATTR_END + 1
CHORD_ATTR_SIZE              = CHORD_ATTR_PAD + 1

# SEMANTIC
SEMANTIC_PAD               = 0.0 

# SCENE_OFFSET
SCENE_OFFSET_PAD        = 0.0 

# MOTION
MOTION_PAD        = 0.0 

# EMOTION
EMOTION_PAD        = 0.0 

# NOTE_DENSITY
NOTE_DENSITY_PAD        = 0.0 

# LOUDNESS
LOUDNESS_PAD        = 0.0 

# OTHER
SEPERATOR               = "========================="
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9
LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32
TORCH_LABEL_TYPE        = torch.long
PREPEND_ZEROS_WIDTH     = 4

# MIDI
TOKEN_END               = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1
VOCAB_SIZE              = TOKEN_PAD + 1
