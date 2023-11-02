import os
import pickle
import random
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from utilities.constants import *
from utilities.device import cpu_device
from utilities.device import get_device

import json

SEQUENCE_START = 0

class VevoDataset(Dataset):
    def __init__(self, dataset_root = "./dataset/", split="train", split_ver="v1", vis_models="2d/clip_l14p", emo_model="6c_l14p", max_seq_chord=300, max_seq_video=300, random_seq=True, is_video = True):
        
        self.dataset_root       = dataset_root

        self.vevo_chord_root = os.path.join( dataset_root, "vevo_chord", "lab_v2_norm", "all")
        self.vevo_emotion_root = os.path.join( dataset_root, "vevo_emotion", emo_model, "all")
        self.vevo_motion_root = os.path.join( dataset_root, "vevo_motion", "all")
        self.vevo_scene_offset_root = os.path.join( dataset_root, "vevo_scene_offset", "all")
        self.vevo_meta_split_path = os.path.join( dataset_root, "vevo_meta", "split", split_ver, split + ".txt")
        
        self.vevo_loudness_root = os.path.join( dataset_root, "vevo_loudness", "all")
        self.vevo_note_density_root = os.path.join( dataset_root, "vevo_note_density", "all")

        self.max_seq_video    = max_seq_video
        self.max_seq_chord    = max_seq_chord
        self.random_seq = random_seq
        self.is_video = is_video

        self.vis_models_arr = vis_models.split(" ")
        self.vevo_semantic_root_list = []
        self.id_list = []

        self.emo_model = emo_model

        if IS_VIDEO:
            for i in range( len(self.vis_models_arr) ):
                p1 = self.vis_models_arr[i].split("/")[0]
                p2 = self.vis_models_arr[i].split("/")[1]
                vevo_semantic_root = os.path.join(dataset_root, "vevo_semantic" , "all" , p1, p2)
                self.vevo_semantic_root_list.append( vevo_semantic_root )
            
        with open( self.vevo_meta_split_path ) as f:
            for line in f:
                self.id_list.append(line.strip())
        
        self.data_files_chord = []      
        self.data_files_emotion = []
        self.data_files_motion = []
        self.data_files_scene_offset = []
        self.data_files_semantic_list = []

        self.data_files_loudness = []
        self.data_files_note_density = []

        for i in range(len(self.vis_models_arr)):
            self.data_files_semantic_list.append([])

        for fid in self.id_list:
            fpath_chord = os.path.join( self.vevo_chord_root, fid + ".lab" )
            fpath_emotion = os.path.join( self.vevo_emotion_root, fid + ".lab" )
            fpath_motion = os.path.join( self.vevo_motion_root, fid + ".lab" )
            fpath_scene_offset = os.path.join( self.vevo_scene_offset_root, fid + ".lab" )

            fpath_loudness = os.path.join( self.vevo_loudness_root, fid + ".lab" )
            fpath_note_density = os.path.join( self.vevo_note_density_root, fid + ".lab" )

            fpath_semantic_list = []
            for vevo_semantic_root in self.vevo_semantic_root_list:
                fpath_semantic = os.path.join( vevo_semantic_root, fid + ".npy" )
                fpath_semantic_list.append(fpath_semantic)
            
            checkFile_semantic = True
            for fpath_semantic in fpath_semantic_list:
                if not os.path.exists(fpath_semantic):
                    checkFile_semantic = False
            
            checkFile_chord = os.path.exists(fpath_chord)
            checkFile_emotion = os.path.exists(fpath_emotion)
            checkFile_motion = os.path.exists(fpath_motion)
            checkFile_scene_offset = os.path.exists(fpath_scene_offset)

            checkFile_loudness = os.path.exists(fpath_loudness)
            checkFile_note_density = os.path.exists(fpath_note_density)

            if checkFile_chord and checkFile_emotion and checkFile_motion \
                and checkFile_scene_offset and checkFile_semantic and checkFile_loudness and checkFile_note_density :

                self.data_files_chord.append(fpath_chord)
                self.data_files_emotion.append(fpath_emotion)
                self.data_files_motion.append(fpath_motion)
                self.data_files_scene_offset.append(fpath_scene_offset)

                self.data_files_loudness.append(fpath_loudness)
                self.data_files_note_density.append(fpath_note_density)

                if IS_VIDEO:
                    for i in range(len(self.vis_models_arr)):
                        self.data_files_semantic_list[i].append( fpath_semantic_list[i] )
        
        chordDicPath = os.path.join( dataset_root, "vevo_meta/chord.json")
        
        chordRootDicPath = os.path.join( dataset_root, "vevo_meta/chord_root.json")
        chordAttrDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr.json")
        
        with open(chordDicPath) as json_file:
            self.chordDic = json.load(json_file)
        
        with open(chordRootDicPath) as json_file:
            self.chordRootDic = json.load(json_file)
        
        with open(chordAttrDicPath) as json_file:
            self.chordAttrDic = json.load(json_file)
        
    def __len__(self):
        return len(self.data_files_chord)

    def __getitem__(self, idx):
        #### ---- CHORD ----- ####
        feature_chord = np.empty(self.max_seq_chord)
        feature_chord.fill(CHORD_PAD)

        feature_chordRoot = np.empty(self.max_seq_chord)
        feature_chordRoot.fill(CHORD_ROOT_PAD)
        feature_chordAttr = np.empty(self.max_seq_chord)
        feature_chordAttr.fill(CHORD_ATTR_PAD)

        key = ""
        with open(self.data_files_chord[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                if line_arr[0] == "key":
                    key = line_arr[1] + " "+ line_arr[2]
                    continue
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break
                chord = line_arr[1]
                chordID = self.chordDic[chord]
                feature_chord[time] = chordID
                chord_arr = chord.split(":")

                if len(chord_arr) == 1:
                    if chord_arr[0] == "N":
                        chordRootID = self.chordRootDic["N"]
                        chordAttrID = self.chordAttrDic["N"]
                        feature_chordRoot[time] = chordRootID
                        feature_chordAttr[time] = chordAttrID
                    else:
                        chordRootID = self.chordRootDic[chord_arr[0]]
                        feature_chordRoot[time] = chordRootID
                        feature_chordAttr[time] = 1
                elif len(chord_arr) == 2:
                    chordRootID = self.chordRootDic[chord_arr[0]]
                    chordAttrID = self.chordAttrDic[chord_arr[1]]
                    feature_chordRoot[time] = chordRootID
                    feature_chordAttr[time] = chordAttrID

        if "major" in key:
            feature_key = torch.tensor([0])
        else:
            feature_key = torch.tensor([1])

        feature_chord = torch.from_numpy(feature_chord)
        feature_chord = feature_chord.to(torch.long)
        
        feature_chordRoot = torch.from_numpy(feature_chordRoot)
        feature_chordRoot = feature_chordRoot.to(torch.long)

        feature_chordAttr = torch.from_numpy(feature_chordAttr)
        feature_chordAttr = feature_chordAttr.to(torch.long)

        feature_key = feature_key.float()
        
        x = feature_chord[:self.max_seq_chord-1]
        tgt = feature_chord[1:self.max_seq_chord]

        x_root = feature_chordRoot[:self.max_seq_chord-1]
        tgt_root = feature_chordRoot[1:self.max_seq_chord]
        x_attr = feature_chordAttr[:self.max_seq_chord-1]
        tgt_attr = feature_chordAttr[1:self.max_seq_chord]

        if time < self.max_seq_chord:
            tgt[time] = CHORD_END
            tgt_root[time] = CHORD_ROOT_END
            tgt_attr[time] = CHORD_ATTR_END
        
        #### ---- SCENE OFFSET ----- ####
        feature_scene_offset = np.empty(self.max_seq_video)
        feature_scene_offset.fill(SCENE_OFFSET_PAD)
        with open(self.data_files_scene_offset[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break
                sceneID = line_arr[1]
                feature_scene_offset[time] = int(sceneID)+1

        feature_scene_offset = torch.from_numpy(feature_scene_offset)
        feature_scene_offset = feature_scene_offset.to(torch.float32)

        #### ---- MOTION ----- ####
        feature_motion = np.empty(self.max_seq_video)
        feature_motion.fill(MOTION_PAD)
        with open(self.data_files_motion[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break
                motion = line_arr[1]
                feature_motion[time] = float(motion)

        feature_motion = torch.from_numpy(feature_motion)
        feature_motion = feature_motion.to(torch.float32)

        #### ---- NOTE_DENSITY ----- ####
        feature_note_density = np.empty(self.max_seq_video)
        feature_note_density.fill(NOTE_DENSITY_PAD)
        with open(self.data_files_note_density[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break
                note_density = line_arr[1]
                feature_note_density[time] = float(note_density)

        feature_note_density = torch.from_numpy(feature_note_density)
        feature_note_density = feature_note_density.to(torch.float32)

        #### ---- LOUDNESS ----- ####
        feature_loudness = np.empty(self.max_seq_video)
        feature_loudness.fill(LOUDNESS_PAD)
        with open(self.data_files_loudness[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break
                loudness = line_arr[1]
                feature_loudness[time] = float(loudness)

        feature_loudness = torch.from_numpy(feature_loudness)
        feature_loudness = feature_loudness.to(torch.float32)

        #### ---- EMOTION ----- ####
        if self.emo_model.startswith("6c"):
            feature_emotion = np.empty( (self.max_seq_video, 6))
        else:
            feature_emotion = np.empty( (self.max_seq_video, 5))

        feature_emotion.fill(EMOTION_PAD)
        with open(self.data_files_emotion[idx], encoding = 'utf-8') as f:
            for line in f:
                line = line.strip()
                line_arr = line.split(" ")
                if line_arr[0] == "time":
                    continue
                time = line_arr[0]
                time = int(time)
                if time >= self.max_seq_chord:
                    break

                if len(line_arr) == 7:
                    emo1, emo2, emo3, emo4, emo5, emo6 = \
                        line_arr[1],line_arr[2],line_arr[3],line_arr[4],line_arr[5],line_arr[6]                    
                    emoList = [ float(emo1), float(emo2), float(emo3), float(emo4), float(emo5), float(emo6) ]
                elif len(line_arr) == 6:
                    emo1, emo2, emo3, emo4, emo5 = \
                        line_arr[1],line_arr[2],line_arr[3],line_arr[4],line_arr[5]
                    emoList = [ float(emo1), float(emo2), float(emo3), float(emo4), float(emo5) ]
                
                emoList = np.array(emoList)
                feature_emotion[time] = emoList

        feature_emotion = torch.from_numpy(feature_emotion)
        feature_emotion = feature_emotion.to(torch.float32)

        feature_emotion_argmax = torch.argmax(feature_emotion, dim=1)
        _, max_prob_indices = torch.max(feature_emotion, dim=1)
        max_prob_values = torch.gather(feature_emotion, dim=1, index=max_prob_indices.unsqueeze(1))
        max_prob_values = max_prob_values.squeeze()

        # -- emotion to chord
        #              maj dim sus4 min7 min sus2 aug dim7 maj6 hdim7 7 min6 maj7
        # 0. extcing : [1,0,1,0,0,0,0,0,0,0,1,0,0]
        # 1. fearful : [0,1,0,1,0,0,0,1,0,1,0,0,0]
        # 2. tense :   [0,1,1,1,0,0,0,0,0,0,1,0,0]
        # 3. sad :     [0,0,0,1,1,1,0,0,0,0,0,0,0]
        # 4. relaxing: [1,0,0,0,0,0,0,0,1,0,0,0,1]
        # 5. neutral : [0,0,0,0,0,0,0,0,0,0,0,0,0]

        a0 = [0]+[1,0,1,0,0,0,0,0,0,0,1,0,0]*12+[0,0]
        a1 = [0]+[0,1,0,1,0,0,0,1,0,1,0,0,0]*12+[0,0]
        a2 = [0]+[0,1,1,1,0,0,0,0,0,0,1,0,0]*12+[0,0]
        a3 = [0]+[0,0,0,1,1,1,0,0,0,0,0,0,0]*12+[0,0]
        a4 = [0]+[1,0,0,0,0,0,0,0,1,0,0,0,1]*12+[0,0]
        a5 = [0]+[0,0,0,0,0,0,0,0,0,0,0,0,0]*12+[0,0]

        aend = [0]+[0,0,0,0,0,0,0,0,0,0,0,0,0]*12+[1,0]
        apad = [0]+[0,0,0,0,0,0,0,0,0,0,0,0,0]*12+[0,1]

        a0_tensor = torch.tensor(a0)
        a1_tensor = torch.tensor(a1)
        a2_tensor = torch.tensor(a2)
        a3_tensor = torch.tensor(a3)
        a4_tensor = torch.tensor(a4)
        a5_tensor = torch.tensor(a5)

        aend_tensor = torch.tensor(aend)
        apad_tensor = torch.tensor(apad)

        mapped_tensor = torch.zeros((300, 159))
        for i, val in enumerate(feature_emotion_argmax):
            if feature_chord[i] == CHORD_PAD:
                mapped_tensor[i] = apad_tensor
            elif feature_chord[i] == CHORD_END:
                mapped_tensor[i] = aend_tensor
            elif val == 0:
                mapped_tensor[i] = a0_tensor
            elif val == 1:
                mapped_tensor[i] = a1_tensor
            elif val == 2:
                mapped_tensor[i] = a2_tensor
            elif val == 3:
                mapped_tensor[i] = a3_tensor
            elif val == 4:
                mapped_tensor[i] = a4_tensor
            elif val == 5:
                mapped_tensor[i] = a5_tensor

        # feature emotion : [1, 300, 6]
        # y : [299, 159]
        # tgt : [299]
        # tgt_emo : [299, 159]
        # tgt_emo_prob : [299]

        tgt_emotion = mapped_tensor[1:]
        tgt_emotion_prob = max_prob_values[1:]
        
        feature_semantic_list = []
        if self.is_video:
            for i in range( len(self.vis_models_arr) ):
                video_feature = np.load(self.data_files_semantic_list[i][idx])
                dim_vf = video_feature.shape[1] # 2048
                video_feature_tensor = torch.from_numpy( video_feature )
                
                feature_semantic = torch.full((self.max_seq_video, dim_vf,), SEMANTIC_PAD , dtype=torch.float32, device=cpu_device())
                if(video_feature_tensor.shape[0] < self.max_seq_video):
                    feature_semantic[:video_feature_tensor.shape[0]] = video_feature_tensor
                else:
                    feature_semantic = video_feature_tensor[:self.max_seq_video]
                feature_semantic_list.append(feature_semantic)

        return { "x":x, 
                "tgt":tgt, 
                "x_root":x_root, 
                "tgt_root":tgt_root, 
                "x_attr":x_attr, 
                "tgt_attr":tgt_attr,
                "semanticList": feature_semantic_list, 
                "key": feature_key,
                "scene_offset": feature_scene_offset,
                "motion": feature_motion,
                "emotion": feature_emotion,
                "tgt_emotion" : tgt_emotion,
                "tgt_emotion_prob" : tgt_emotion_prob,
                "note_density" : feature_note_density,
                "loudness" : feature_loudness
                }

def create_vevo_datasets(dataset_root = "./dataset", max_seq_chord=300, max_seq_video=300, vis_models="2d/clip_l14p", emo_model="6c_l14p", split_ver="v1", random_seq=True, is_video=True):

    train_dataset = VevoDataset(
        dataset_root = dataset_root, split="train", split_ver=split_ver, 
        vis_models=vis_models, emo_model =emo_model, max_seq_chord=max_seq_chord, max_seq_video=max_seq_video, 
        random_seq=random_seq, is_video = is_video )
    
    val_dataset = VevoDataset(
        dataset_root = dataset_root, split="val", split_ver=split_ver, 
        vis_models=vis_models, emo_model =emo_model, max_seq_chord=max_seq_chord, max_seq_video=max_seq_video, 
        random_seq=random_seq, is_video = is_video )
    
    test_dataset = VevoDataset(
        dataset_root = dataset_root, split="test", split_ver=split_ver, 
        vis_models=vis_models, emo_model =emo_model, max_seq_chord=max_seq_chord, max_seq_video=max_seq_video, 
        random_seq=random_seq, is_video = is_video )
    
    return train_dataset, val_dataset, test_dataset

def compute_vevo_accuracy(out, tgt):
    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != CHORD_PAD)

    out = out[mask]
    tgt = tgt[mask]

    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc

def compute_hits_k(out, tgt, k):
    softmax = nn.Softmax(dim=-1)
    out = softmax(out)
    _, topk_indices = torch.topk(out, k, dim=-1)  # Get the indices of top-k values

    tgt = tgt.flatten()

    topk_indices = torch.squeeze(topk_indices, dim = 0)

    num_right = 0 
    pt = 0
    for i, tlist in enumerate(topk_indices):
        if tgt[i] == CHORD_PAD:
            num_right += 0
        else:
            pt += 1 
            if tgt[i].item() in tlist:
                num_right += 1

    # Empty
    if len(tgt) == 0:
        return 1.0
    
    num_right = torch.tensor(num_right, dtype=torch.float32)
    hitk = num_right / pt

    return hitk

def compute_hits_k_root_attr(out_root, out_attr, tgt, k):
    softmax = nn.Softmax(dim=-1)
    out_root = softmax(out_root)
    out_attr = softmax(out_attr)

    tensor_shape = torch.Size([1, 299, 159])
    out = torch.zeros(tensor_shape)
    for i in range(out.shape[-1]):
        if i == 0 :
            out[0, :, i] = out_root[0, :, 0] * out_attr[0, :, 0] 
        elif i == 157:
            out[0, :, i] = out_root[0, :, 13] * out_attr[0, :, 14]
        elif i == 158:
            out[0, :, i] = out_root[0, :, 14] * out_attr[0, :, 15]
        else:
            rootindex =  int( (i-1)/13 ) + 1
            attrindex =  (i-1)%13 + 1
            out[0, :, i] = out_root[0, :, rootindex] * out_attr[0, :, attrindex]

    out = softmax(out)
    _, topk_indices = torch.topk(out, k, dim=-1)  # Get the indices of top-k values

    tgt = tgt.flatten()

    topk_indices = torch.squeeze(topk_indices, dim = 0)

    num_right = 0 
    pt = 0
    for i, tlist in enumerate(topk_indices):
        if tgt[i] == CHORD_PAD:
            num_right += 0
        else:
            pt += 1 
            if tgt[i].item() in tlist:
                num_right += 1

    if len(tgt) == 0:
        return 1.0
    
    num_right = torch.tensor(num_right, dtype=torch.float32)
    hitk = num_right / pt

    return hitk

def compute_vevo_correspondence(out, tgt, tgt_emotion, tgt_emotion_prob, emotion_threshold):

    tgt_emotion = tgt_emotion.squeeze()
    tgt_emotion_prob = tgt_emotion_prob.squeeze()

    dataset_root = "./dataset/"
    chordRootInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_root_inv.json")
    chordAttrInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr_inv.json")
    chordAttrDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr.json")
    
    chordDicPath = os.path.join( dataset_root, "vevo_meta/chord.json")
    chordInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_inv.json")

    with open(chordRootInvDicPath) as json_file:
        chordRootInvDic = json.load(json_file)
    with open(chordAttrDicPath) as json_file:
        chordAttrDic = json.load(json_file)
    with open(chordAttrInvDicPath) as json_file:
        chordAttrInvDic = json.load(json_file)
    with open(chordDicPath) as json_file:
        chordDic = json.load(json_file)
    with open(chordInvDicPath) as json_file:
        chordInvDic = json.load(json_file)

    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)
    out = out.flatten()

    tgt = tgt.flatten()

    num_right = 0
    tgt_emotion_quality = tgt_emotion[:, 0:14]
    pt = 0 
    for i, out_element in enumerate( out ):

        all_zeros = torch.all(tgt_emotion_quality[i] == 0)
        if tgt_emotion[i][-1] == 1 or all_zeros or tgt_emotion_prob[i] < emotion_threshold:
            num_right += 0
        else:
            pt += 1
            if out_element.item() != CHORD_END and out_element.item() != CHORD_PAD:
                gen_chord = chordInvDic[ str( out_element.item() ) ]

                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    out_quality = 1
                elif len(chord_arr) == 2:
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    out_quality = chordAttrID # 0:N, 1:maj ... 13:maj7

                if tgt_emotion_quality[i][out_quality] == 1:
                    num_right += 1
                    

    if(len(tgt_emotion) == 0):
        return 1.0
    
    if(pt == 0):
        return -1
    
    num_right = torch.tensor(num_right, dtype=torch.float32)
    acc = num_right / pt

    return acc

def compute_vevo_correspondence_root_attr(y_root, y_attr, tgt, tgt_emotion, tgt_emotion_prob, emotion_threshold):

    tgt_emotion = tgt_emotion.squeeze()
    tgt_emotion_prob = tgt_emotion_prob.squeeze()

    dataset_root = "./dataset/"
    chordRootInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_root_inv.json")
    chordAttrInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr_inv.json")
    chordAttrDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr.json")
    
    chordDicPath = os.path.join( dataset_root, "vevo_meta/chord.json")
    chordInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_inv.json")

    with open(chordRootInvDicPath) as json_file:
        chordRootInvDic = json.load(json_file)
    with open(chordAttrDicPath) as json_file:
        chordAttrDic = json.load(json_file)
    with open(chordAttrInvDicPath) as json_file:
        chordAttrInvDic = json.load(json_file)
    with open(chordDicPath) as json_file:
        chordDic = json.load(json_file)
    with open(chordInvDicPath) as json_file:
        chordInvDic = json.load(json_file)

    softmax = nn.Softmax(dim=-1)

    y_root = torch.argmax(softmax(y_root), dim=-1)
    y_attr = torch.argmax(softmax(y_attr), dim=-1)
    
    y_root = y_root.flatten()
    y_attr = y_attr.flatten()

    tgt = tgt.flatten()
    y = np.empty( len(tgt) )

    y.fill(CHORD_PAD)

    for i in range(len(tgt)):
        if y_root[i].item() == CHORD_ROOT_PAD or y_attr[i].item() == CHORD_ATTR_PAD:
            y[i] = CHORD_PAD
        elif y_root[i].item() == CHORD_ROOT_END or y_attr[i].item() == CHORD_ATTR_END:
            y[i] = CHORD_END
        else:
            chordRoot = chordRootInvDic[str(y_root[i].item())]
            chordAttr = chordAttrInvDic[str(y_attr[i].item())]
            if chordRoot == "N":
                y[i] = 0
            else:
                if chordAttr == "N" or chordAttr == "maj":
                    y[i] = chordDic[chordRoot]
                else:
                    chord = chordRoot + ":" + chordAttr
                    y[i] = chordDic[chord]

    y = torch.from_numpy(y)
    y = y.to(torch.long)
    y = y.to(get_device())
    y = y.flatten()

    num_right = 0
    tgt_emotion_quality = tgt_emotion[:, 0:14]
    pt = 0 
    for i, y_element in enumerate( y ):
        all_zeros = torch.all(tgt_emotion_quality[i] == 0)
        if tgt_emotion[i][-1] == 1 or all_zeros or tgt_emotion_prob[i] < emotion_threshold:
            num_right += 0
        else:
            pt += 1
            if y_element.item() != CHORD_END and y_element.item() != CHORD_PAD:
                gen_chord = chordInvDic[ str( y_element.item() ) ]
                chord_arr = gen_chord.split(":")
                if len(chord_arr) == 1:
                    y_quality = 1
                elif len(chord_arr) == 2:
                    chordAttrID = chordAttrDic[chord_arr[1]]
                    y_quality = chordAttrID # 0:N, 1:maj ... 13:maj7

                if tgt_emotion_quality[i][y_quality] == 1:
                    num_right += 1
                    
    if(len(tgt_emotion) == 0):
        return 1.0
    
    if(pt == 0):
        return -1
    
    num_right = torch.tensor(num_right, dtype=torch.float32)
    acc = num_right / pt
    return acc

def compute_vevo_accuracy_root_attr(y_root, y_attr, tgt):

    dataset_root = "./dataset/"
    chordRootInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_root_inv.json")
    chordAttrInvDicPath = os.path.join( dataset_root, "vevo_meta/chord_attr_inv.json")
    chordDicPath = os.path.join( dataset_root, "vevo_meta/chord.json")
    
    with open(chordRootInvDicPath) as json_file:
        chordRootInvDic = json.load(json_file)
    with open(chordAttrInvDicPath) as json_file:
        chordAttrInvDic = json.load(json_file)
    with open(chordDicPath) as json_file:
        chordDic = json.load(json_file)

    softmax = nn.Softmax(dim=-1)

    y_root = torch.argmax(softmax(y_root), dim=-1)
    y_attr = torch.argmax(softmax(y_attr), dim=-1)
    
    y_root = y_root.flatten()
    y_attr = y_attr.flatten()

    tgt = tgt.flatten()

    mask = (tgt != CHORD_PAD)
    y = np.empty( len(tgt) )
    y.fill(CHORD_PAD)

    for i in range(len(tgt)):
        if y_root[i].item() == CHORD_ROOT_PAD or y_attr[i].item() == CHORD_ATTR_PAD:
            y[i] = CHORD_PAD
        elif y_root[i].item() == CHORD_ROOT_END or y_attr[i].item() == CHORD_ATTR_END:
            y[i] = CHORD_END
        else:
            chordRoot = chordRootInvDic[str(y_root[i].item())]
            chordAttr = chordAttrInvDic[str(y_attr[i].item())]
            if chordRoot == "N":
                y[i] = 0
            else:
                if chordAttr == "N" or chordAttr == "maj":
                    y[i] = chordDic[chordRoot]
                else:
                    chord = chordRoot + ":" + chordAttr
                    y[i] = chordDic[chord]

    y = torch.from_numpy(y)
    y = y.to(torch.long)
    y = y.to(get_device())

    y = y[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    num_right = (y == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc

