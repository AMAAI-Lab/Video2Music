import gradio as gr
from pathlib import Path

import torch
import shutil
import os
import subprocess
import cv2
import math
import clip
import numpy as np
from PIL import Image
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg
from scenedetect.scene_manager import save_images
from utilities.constants import *
from utilities.chord_to_midi import *

from model.video_music_transformer import VideoMusicTransformer
from model.video_regression import VideoRegression

import json
from midi2audio import FluidSynth
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import random
from moviepy.editor import *
import time

from tqdm import tqdm
from huggingface_hub import snapshot_download

from gradio import Markdown

from pytube import YouTube

all_key_names = ['C major', 'G major', 'D major', 'A major',
                 'E major', 'B major', 'F major', 'Bb major',
                 'Eb major', 'Ab major', 'Db major', 'Gb major',
                 'A minor', 'E minor', 'B minor', 'F# minor',
                 'C# minor', 'G# minor', 'D minor', 'G minor',
                 'C minor', 'F minor', 'Bb minor', 'Eb minor',
                 ]

traspose_key_dic = {
    'F major' : -7,
    'Gb major' : -6,
    'G major' : -5,
    'Ab major' : -4,
    'A major' : -3,
    'Bb major' : -2,
    'B major' : -1,
    'C major' : 0,
    'Db major' : 1,
    'D major' : 2,
    'Eb major' : 3,
    'E major' : 4,
    'D minor' : -7,
    'Eb minor' : -6,
    'E minor' : -5,
    'F minor' : -4,
    'F# minor' : -3,
    'G minor' : -2,
    'G# minor' : -1,
    'A minor' : 0,
    'Bb minor' : 1,
    'B minor' : 2,
    'C minor' : 3,
    'C# minor' : 4
}

flatsharpDic = {
    'Db':'C#', 
    'Eb':'D#', 
    'Gb':'F#', 
    'Ab':'G#', 
    'Bb':'A#'
}

max_conseq_N = 0
max_conseq_chord = 2
tempo = 120
duration = 2

min_loudness = 0  # Minimum loudness level in the input range
max_loudness = 50  # Maximum loudness level in the input range
min_velocity = 49  # Minimum velocity value in the output range
max_velocity = 112  # Maximum velocity value in the output range


def split_video_into_frames(video, frame_dir):
    output_path = os.path.join(frame_dir, f"%03d.jpg")
    cmd = f"ffmpeg -i {video} -vf \"select=bitor(gte(t-prev_selected_t\,1)\,isnan(prev_selected_t))\" -vsync 0 -qmin 1 -q:v 1 {output_path}"        
    subprocess.call(cmd, shell=True)

def gen_semantic_feature(frame_dir, semantic_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    file_names = os.listdir(frame_dir)
    sorted_file_names = sorted(file_names)

    output_path = semantic_dir / "semantic.npy"
    if torch.cuda.is_available():
        features = torch.FloatTensor(len(sorted_file_names), 768).fill_(0)
        features = features.to(device)
        
        for idx, file_name in enumerate(sorted_file_names):
            fpath = frame_dir / file_name
            image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)          
            with torch.no_grad():
                image_features = model.encode_image(image)
            features[idx] = image_features[0]
        features = features.cpu().numpy()
        np.save(output_path, features)
    else:
        features = torch.FloatTensor(len(sorted_file_names), 768).fill_(0)
        for idx, file_name in enumerate(sorted_file_names):
            fpath = frame_dir / file_name
            image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)          
            with torch.no_grad():
                image_features = model.encode_image(image)
            features[idx] = image_features[0]
        features = features.numpy()
        np.save(output_path, features)

def gen_emotion_feature(frame_dir, emotion_dir):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    text = clip.tokenize(["exciting", "fearful", "tense", "sad", "relaxing", "neutral"]).to(device)

    file_names = os.listdir(frame_dir)
    sorted_file_names = sorted(file_names)
    output_path = emotion_dir / "emotion.lab" 

    emolist = []
    for file_name in sorted_file_names:
        fpath = frame_dir / file_name
        image = preprocess(Image.open(fpath)).unsqueeze(0).to(device)                
        with torch.no_grad():  
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        fp1 = format(probs[0][0], ".4f")
        fp2 = format(probs[0][1], ".4f")
        fp3 = format(probs[0][2], ".4f")
        fp4 = format(probs[0][3], ".4f")
        fp5 = format(probs[0][4], ".4f")
        fp6 = format(probs[0][5], ".4f")
        
        emo_val = str(fp1) +" "+ str(fp2) +" "+ str(fp3) +" "+ str(fp4) +" "+ str(fp5) + " " + str(fp6)
        emolist.append(emo_val)
    
    with open(output_path ,'w' ,encoding = 'utf-8') as f:
        f.write("time exciting_prob fearful_prob tense_prob sad_prob relaxing_prob neutral_prob\n")
        for i in range(0, len(emolist) ):
            f.write(str(i) + " "+emolist[i]+"\n")


def gen_scene_feature(video, scene_dir, frame_dir):
    video_stream = open_video(str(video))
    
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video_stream, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    sec = 0
    scenedict = {}
    for idx, scene in enumerate(scene_list):
        end_int = math.ceil(scene[1].get_seconds())
        for s in range (sec, end_int):
            scenedict[s] = str(idx)
            sec += 1
    
    fpathname = scene_dir / "scene.lab"

    if len(scene_list) == 0:
        fsize = len( os.listdir(frame_dir) )
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, fsize):
                f.write(str(i) + " "+"0"+"\n")
    else:
        with open(fpathname,'w',encoding = 'utf-8') as f:
            for i in range(0, len(scenedict)):
                f.write(str(i) + " "+scenedict[i]+"\n")

def gen_scene_offset_feature(scene_dir, scene_offset_dir):
    src = scene_dir / "scene.lab"
    tgt = scene_offset_dir / "scene_offset.lab"
    
    id_list = []
    with open(src, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            if len(line_arr) == 2 :
                time = int(line_arr[0])
                scene_id = int(line_arr[1])
                id_list.append(scene_id)

    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    
    with open(tgt,'w',encoding = 'utf-8') as f:
        for i in range(0, len(offset_list)):
            f.write(str(i) + " " + str(offset_list[i]) + "\n")

def gen_motion_feature(video, motion_dir):
    cap = cv2.VideoCapture(str(video))
    prev_frame = None
    prev_time = 0
    motion_value = 0
    motiondict = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        motiondict[0] = "0.0000"
        if prev_frame is not None and curr_time - prev_time >= 1:
            diff = cv2.absdiff(frame, prev_frame)
            diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
            motion_value = diff_rgb.mean()
            motion_value = format(motion_value, ".4f")
            motiondict[int(curr_time)] = str(motion_value)
            prev_time = int(curr_time)
        prev_frame = frame.copy()
    cap.release()
    cv2.destroyAllWindows()
    fpathname = motion_dir / "motion.lab"
    
    with open(fpathname,'w',encoding = 'utf-8') as f:
        for i in range(0, len(motiondict)):
            f.write(str(i) + " "+motiondict[i]+"\n")


# def get_motion_feature(scene_dir, scene_offset_dir):
# fpath_emotion = emotion_dir / "emotion.lab" 
# fpath_motion = motion_dir / "motion.lab" 

def get_scene_offset_feature(scene_offset_dir, max_seq_chord=300, max_seq_video=300):
    feature_scene_offset = np.empty(max_seq_video)
    feature_scene_offset.fill(SCENE_OFFSET_PAD)
    fpath_scene_offset = scene_offset_dir / "scene_offset.lab" 

    with open(fpath_scene_offset, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            time = line_arr[0]
            time = int(time)
            if time >= max_seq_chord:
                break
            sceneID = line_arr[1]
            feature_scene_offset[time] = int(sceneID)+1

    feature_scene_offset = torch.from_numpy(feature_scene_offset)
    feature_scene_offset = feature_scene_offset.to(torch.float32)

    return feature_scene_offset

def get_motion_feature(motion_dir, max_seq_chord=300, max_seq_video=300):
    fpath_motion = motion_dir / "motion.lab" 
    feature_motion = np.empty(max_seq_video)
    feature_motion.fill(MOTION_PAD)
    with open(fpath_motion, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            time = line_arr[0]
            time = int(time)
            if time >= max_seq_chord:
                break
            motion = line_arr[1]
            feature_motion[time] = float(motion)

    feature_motion = torch.from_numpy(feature_motion)
    feature_motion = feature_motion.to(torch.float32)
    return feature_motion

def get_emotion_feature(emotion_dir, max_seq_chord=300, max_seq_video=300):
    fpath_emotion = emotion_dir / "emotion.lab" 
    feature_emotion = np.empty((max_seq_video, 6))
    feature_emotion.fill(EMOTION_PAD)

    with open(fpath_emotion, encoding = 'utf-8') as f:
        for line in f:
            line = line.strip()
            line_arr = line.split(" ")
            if line_arr[0] == "time":
                continue
            time = line_arr[0]
            time = int(time)
            if time >= max_seq_chord:
                break
            emo1, emo2, emo3, emo4, emo5, emo6 = \
                line_arr[1],line_arr[2],line_arr[3],line_arr[4],line_arr[5],line_arr[6]                    
            emoList = [ float(emo1), float(emo2), float(emo3), float(emo4), float(emo5), float(emo6) ]
            emoList = np.array(emoList)
            feature_emotion[time] = emoList

    feature_emotion = torch.from_numpy(feature_emotion)
    feature_emotion = feature_emotion.to(torch.float32)
    return feature_emotion

def get_semantic_feature(semantic_dir, max_seq_chord=300, max_seq_video=300):
    fpath_semantic = semantic_dir / "semantic.npy" 
    
    video_feature = np.load(fpath_semantic)
    dim_vf = video_feature.shape[1]

    video_feature_tensor = torch.from_numpy( video_feature )
    feature_semantic = torch.full((max_seq_video, dim_vf,), SEMANTIC_PAD , dtype=torch.float32, device=torch.device("cpu"))

    if(video_feature_tensor.shape[0] < max_seq_video):
        feature_semantic[:video_feature_tensor.shape[0]] = video_feature_tensor
    else:
        feature_semantic = video_feature_tensor[:max_seq_video]
    
    return feature_semantic


def text_clip(text: str, duration: int, start_time: int = 0):
    t = TextClip(text, font='Georgia-Regular', fontsize=24, color='white')
    t = t.set_position(("center", 20)).set_duration(duration)
    t = t.set_start(start_time)
    return t

def convert_format_id_to_offset(id_list):
    offset_list = []
    current_id = id_list[0]
    offset = 0
    for i in range(len(id_list)):
        if id_list[i] != current_id:
            current_id = id_list[i]
            offset = 0
        offset_list.append(offset)
        offset += 1
    return offset_list


class Video2music:
    def __init__(
        self,
        name="amaai-lab/video2music",
        device="cuda:0",
        cache_dir=None,
        local_files_only=False,
    ):
        # path = snapshot_download(repo_id=name, cache_dir=cache_dir)

        self.device = device
        
        # self.model.device = device
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # f"{path}/beats/microsoft-deberta-v3-large.pt"

        # self.model_weights = f"{path}/saved_models/AMT/best_loss_weights.pickle"
        # self.modelReg_weights = f"{path}/saved_models/AMT/best_rmse_weights.pickle"

        self.model_weights = "saved_models/AMT/best_loss_weights.pickle"
        self.modelReg_weights = "saved_models/AMT/best_rmse_weights.pickle"

        self.total_vf_dim = 776
        # 768 (sem) + 1 (mo) + 1 (scene) + 6 (emo)
        self.max_seq_video = 300
        self.max_seq_chord = 300
        
        self.model = VideoMusicTransformer(n_layers=6, num_heads=8,
                    d_model=512, dim_feedforward=1024,
                    max_sequence_midi=2048, max_sequence_video=300, 
                    max_sequence_chord=300, total_vf_dim=self.total_vf_dim, rpr=RPR).to(device)
        
        self.model.load_state_dict(torch.load(self.model_weights, map_location=device))
        self.modelReg = VideoRegression(max_sequence_video=300, total_vf_dim=self.total_vf_dim, regModel= "bigru").to(device)
        self.modelReg.load_state_dict(torch.load(self.modelReg_weights, map_location=device))

        self.model.eval()
        self.modelReg.eval()

        self.SF2_FILE = "default_sound_font.sf2"

    def generate(self, video, primer, key):

        feature_dir = Path("./feature")
        output_dir = Path("./output")
        if feature_dir.exists():
            shutil.rmtree(str(feature_dir))
        if output_dir.exists():
            shutil.rmtree(str(output_dir))
        
        feature_dir.mkdir(parents=True)
        output_dir.mkdir(parents=True)
        
        frame_dir = feature_dir / "vevo_frame"

        #video features
        semantic_dir = feature_dir / "vevo_semantic"
        emotion_dir = feature_dir / "vevo_emotion"
        scene_dir = feature_dir / "vevo_scene"
        scene_offset_dir = feature_dir / "vevo_scene_offset"
        motion_dir = feature_dir / "vevo_motion"

        frame_dir.mkdir(parents=True)
        semantic_dir.mkdir(parents=True)
        emotion_dir.mkdir(parents=True)
        scene_dir.mkdir(parents=True)
        scene_offset_dir.mkdir(parents=True)
        motion_dir.mkdir(parents=True)
        
        #music features
        chord_dir = feature_dir / "vevo_chord"
        loudness_dir = feature_dir / "vevo_loudness"
        note_density_dir = feature_dir / "vevo_note_density"
        
        chord_dir.mkdir(parents=True)
        loudness_dir.mkdir(parents=True)
        note_density_dir.mkdir(parents=True)

        split_video_into_frames(video, frame_dir)
        gen_semantic_feature(frame_dir, semantic_dir)
        gen_emotion_feature(frame_dir, emotion_dir)
        gen_scene_feature(video, scene_dir, frame_dir)
        gen_scene_offset_feature(scene_dir, scene_offset_dir)
        gen_motion_feature(video, motion_dir)

        feature_scene_offset = get_scene_offset_feature(scene_offset_dir)
        feature_motion = get_motion_feature(motion_dir)
        feature_emotion = get_emotion_feature(emotion_dir)
        feature_semantic = get_semantic_feature(semantic_dir)

        # cuda
        feature_scene_offset = feature_scene_offset.to(self.device)
        feature_motion = feature_motion.to(self.device)
        feature_emotion = feature_emotion.to(self.device)

        feature_scene_offset = feature_scene_offset.unsqueeze(0)
        feature_motion = feature_motion.unsqueeze(0)
        feature_emotion = feature_emotion.unsqueeze(0)

        feature_semantic = feature_semantic.to(self.device)
        feature_semantic_list = []
        feature_semantic = torch.unsqueeze(feature_semantic, 0)
        feature_semantic_list.append( feature_semantic.to(self.device) )
        #feature_semantic_list.append( feature_semantic )

        if "major" in key:
            feature_key = torch.tensor([0])
            feature_key = feature_key.float()
        elif "minor" in key:
            feature_key = torch.tensor([1])
            feature_key = feature_key.float()
        
        feature_key = feature_key.to(self.device)

        with open('dataset/vevo_meta/chord.json') as json_file:
            chordDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_inv.json') as json_file:
            chordInvDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_root.json') as json_file:
            chordRootDic = json.load(json_file)
        with open('dataset/vevo_meta/chord_attr.json') as json_file:
            chordAttrDic = json.load(json_file)

        primer = primer.strip()
        if primer.strip() == "":
            if "major" in key:
                primer = "C"
            else:
                primer = "Am"
        
        pChordList = primer.split()

        primerCID = []
        primerCID_root = []
        primerCID_attr = []
        
        for pChord in pChordList:
            if len(pChord) > 1:
                if pChord[1] == "b":
                    pChord = flatsharpDic [ pChord[0:2] ] + pChord[2:]
                type_idx = 0
                if pChord[1] == "#":
                    pChord = pChord[0:2] + ":" + pChord[2:]
                    type_idx = 2
                else:
                    pChord = pChord[0:1] + ":" + pChord[1:]
                    type_idx = 1
                if pChord[type_idx+1:] == "m":
                    pChord = pChord[0:type_idx] + ":min"
                if pChord[type_idx+1:] == "m6":
                    pChord = pChord[0:type_idx] + ":min6"
                if pChord[type_idx+1:] == "m7":
                    pChord = pChord[0:type_idx] + ":min7"
                if pChord[type_idx+1:] == "M6":
                    pChord = pChord[0:type_idx] + ":maj6"
                if pChord[type_idx+1:] == "M7":
                    pChord = pChord[0:type_idx] + ":maj7"
                if pChord[type_idx+1:] == "":
                    pChord = pChord[0:type_idx]
            
            print("pchord is ", pChord)
            chordID = chordDic[pChord]
            primerCID.append(chordID)
            
            chord_arr = pChord.split(":")
            if len(chord_arr) == 1:
                chordRootID = chordRootDic[chord_arr[0]]
                primerCID_root.append(chordRootID)
                primerCID_attr.append(0)
            elif len(chord_arr) == 2:
                chordRootID = chordRootDic[chord_arr[0]]
                chordAttrID = chordAttrDic[chord_arr[1]]
                primerCID_root.append(chordRootID)
                primerCID_attr.append(chordAttrID)
        
        primerCID = np.array(primerCID)
        primerCID = torch.from_numpy(primerCID)
        primerCID = primerCID.to(torch.long)
        primerCID = primerCID.to(self.device)

        primerCID_root = np.array(primerCID_root)
        primerCID_root = torch.from_numpy(primerCID_root)
        primerCID_root = primerCID_root.to(torch.long)
        primerCID_root = primerCID_root.to(self.device)
        
        primerCID_attr = np.array(primerCID_attr)
        primerCID_attr = torch.from_numpy(primerCID_attr)
        primerCID_attr = primerCID_attr.to(torch.long)
        primerCID_attr = primerCID_attr.to(self.device)

        # self.model.eval()
        # self.modelReg.eval()

        with torch.set_grad_enabled(False):
            rand_seq = self.model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = 300, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord)
            
            y = self.modelReg(
                        feature_semantic_list, 
                        feature_scene_offset,
                        feature_motion,
                        feature_emotion)
        
            y   = y.reshape(y.shape[0] * y.shape[1], -1)

            y_note_density, y_loudness = torch.split(y, split_size_or_sections=1, dim=1)
            y_note_density_np = y_note_density.cpu().numpy()
            y_note_density_np = np.round(y_note_density_np).astype(int)
            y_note_density_np = np.clip(y_note_density_np, 0, 40)

            y_loudness_np = y_loudness.cpu().numpy()
            y_loudness_np_lv = (y_loudness_np * 100).astype(int)
            y_loudness_np_lv = np.clip(y_loudness_np_lv, 0, 50)
            velolistExp = []
            exponent = 0.3
            for item in y_loudness_np_lv:
                loudness = item[0]
                velocity_exp = np.round(((loudness - min_loudness) / (max_loudness - min_loudness)) ** exponent * (max_velocity - min_velocity) + min_velocity)
                velocity_exp = int(velocity_exp)
                velolistExp.append(velocity_exp)
            
            densitylist = []
            for item in y_loudness_np_lv:
                density = item[0]
                if density <= 6:
                    densitylist.append(0)
                elif density <= 12:
                    densitylist.append(1)
                elif density <= 18:
                    densitylist.append(2)
                elif density <= 24:
                    densitylist.append(3)
                else:
                    densitylist.append(4)
            
            # generated ChordID to ChordSymbol
            chord_genlist = []
            chordID_genlist= rand_seq[0].cpu().numpy()
            for i in chordID_genlist:
                chord_genlist.append(chordInvDic[str(i)])
            
            chord_offsetlist = convert_format_id_to_offset(chord_genlist)
            f_path_midi = output_dir / "output.mid"
            f_path_flac = output_dir / "output.flac"
            f_path_video_out = output_dir / "output.mp4"

            # ChordSymbol to MIDI file with voicing
            MIDI = MIDIFile(1)
            MIDI.addTempo(0, 0, tempo)
            midi_chords_orginal = []
            for i, k in enumerate(chord_genlist):
                k = k.replace(":", "")
                if k == "N":
                    midi_chords_orginal.append([])
                else:
                    midi_chords_orginal.append(Chord(k).getMIDI("c", 4))
            midi_chords = voice(midi_chords_orginal)
            trans = traspose_key_dic[key]

            for i, chord in enumerate(midi_chords):
                if densitylist[i] == 0:
                    if len(chord) >= 4:
                        if chord_offsetlist[i] % 2 == 0:
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                        else:
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                elif densitylist[i] == 1:
                    if len(chord) >= 4:
                        if chord_offsetlist[i] % 2 == 0:
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                        else:
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                elif densitylist[i] == 2:
                    if len(chord) >= 4:
                        if chord_offsetlist[i] % 2 == 0:
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                        else:
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                elif densitylist[i] == 3:
                    if len(chord) >= 4:
                        if chord_offsetlist[i] % 2 == 0:
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.75 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                        else:
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0.75 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                elif densitylist[i] == 4:
                    if len(chord) >= 4:
                        if chord_offsetlist[i] % 2 == 0:
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.75 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.75 ,  duration,  velolistExp[i])
                        else:
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[0]+trans,  i * duration + 0.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 0.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 0.75 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[3]+trans,  i * duration + 1 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.25 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[1]+trans,  i * duration + 1.5 ,  duration,  velolistExp[i])
                            MIDI.addNote(0, 0, chord[2]+trans,  i * duration + 1.75 ,  duration,  velolistExp[i])
            
            with open(f_path_midi, "wb") as outputFile:
                MIDI.writeFile(outputFile)
            
            # Convert midi to audio (e.g., flac)
            fs = FluidSynth(sound_font=self.SF2_FILE)
            fs.midi_to_audio(str(f_path_midi), str(f_path_flac))

            # Render generated music into input video
            audio_mp = mp.AudioFileClip(str(f_path_flac))
            video_mp = mp.VideoFileClip(str(video))

            audio_mp = audio_mp.subclip(0, video_mp.duration )
            final = video_mp.set_audio(audio_mp)

            final.write_videofile(str(f_path_video_out), 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            )
            return Path(str(f_path_video_out))