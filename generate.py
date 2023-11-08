import torch
import torch.nn as nn
import os
import random

from third_party.midi_processor.processor import decode_midi, encode_midi
from utilities.argument_funcs import parse_generate_args, print_generate_args
from utilities.chord_to_midi import *

from model.music_transformer import MusicTransformer
from model.video_music_transformer import VideoMusicTransformer
from model.video_regression import VideoRegression

from dataset.vevo_dataset import compute_vevo_accuracy, create_vevo_datasets

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam

from utilities.constants import *
from utilities.device import get_device, use_cuda
import numpy as np
import json

from midi2audio import FluidSynth

import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os, random, shutil
from moviepy.editor import *
import time

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver
test_id = "049"
num_prime_chord = 30

is_voice = True
isArp = True
duration = 2
tempo = 120
octave = 4
velocity = 100
subdivide = 1
key = "c"
isPrimer = False

min_loudness = 0  # Minimum loudness level in the input range
max_loudness = 50  # Maximum loudness level in the input range
min_velocity = 49  # Minimum velocity value in the output range
max_velocity = 112  # Maximum velocity value in the output range

custumPrimer = ["C","Am","Dm","G"]
custumKey = "" 
#minor or major

flatsharpDic = {
    'Db':'C#', 
    'Eb':'D#', 
    'Gb':'F#', 
    'Ab':'G#', 
    'Bb':'A#'
}

regModel = "bigru"

max_conseq_N = 0
max_conseq_chord = 2

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

def main():
    testFileList=[]
    valFileList=[]
    
    with open('dataset/vevo_meta/split/'+ split_ver +'/test.txt') as txt_file:
        for line in txt_file:
            testFileList.append(line.strip())
    with open('dataset/vevo_meta/split/'+ split_ver +'/val.txt') as txt_file:
        for line in txt_file:
            valFileList.append(line.strip())
    
    with open('dataset/vevo_meta/chord.json') as json_file:
        chordDic = json.load(json_file)
    with open('dataset/vevo_meta/chord_inv.json') as json_file:
        chordInvDic = json.load(json_file)

    with open('dataset/vevo_meta/chord_root.json') as json_file:
        chordRootDic = json.load(json_file)
    with open('dataset/vevo_meta/chord_attr.json') as json_file:
        chordAttrDic = json.load(json_file)


    args = parse_generate_args()

    args.test_id = test_id
    args.num_prime_chord = num_prime_chord

    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)
    

    _, val_dataset, test_dataset = create_vevo_datasets(
            dataset_root = "./dataset/", 
            max_seq_chord = args.max_sequence_chord, 
            max_seq_video = args.max_sequence_video, 
            vis_models = args.vis_models,
            emo_model = args.emo_model, 
            split_ver = SPLIT_VER, 
            random_seq = False, 
            is_video = args.is_video)
    
    if test_id in testFileList:
        dataset = test_dataset
    elif test_id in valFileList:
        dataset = val_dataset
    else:
        assert False, f"Test id: {args.test_id} not in test or val dataset"
    
    total_vf_dim = 0
    if args.is_video:
        for vf in dataset[0]["semanticList"]:
            total_vf_dim += vf.shape[1]
    
    total_vf_dim += 1 # Scene_offset
    total_vf_dim += 1 # Motion
    
    # Emotion
    if args.emo_model.startswith("6c"):
        total_vf_dim += 6
    else:
        total_vf_dim += 5
    
    test_id_idx = -1
    if(args.test_id is None):
        test_id_idx = int(random.randrange(len(dataset)))
    else:
        test_id_idx = -1
        for i in range( len(dataset) ):
            if int(args.test_id) == int( dataset.data_files_chord[i].split("/")[-1][:3] ):
                test_id_idx = i
        if test_id_idx == -1:
            assert False, f"Test id: {args.test_id} not in test dataset"

    primer = dataset[test_id_idx]["x"].to(get_device())
    primer_root = dataset[test_id_idx]["x_root"].to(get_device())
    primer_attr = dataset[test_id_idx]["x_attr"].to(get_device())
    
    feature_semantic_list = [] 
    for feature_semantic in dataset[test_id_idx]["semanticList"]:
        feature_semantic = torch.unsqueeze(feature_semantic, 0)
        feature_semantic_list.append( feature_semantic.to(get_device()) )

    feature_scene_offset = dataset[test_id_idx]["scene_offset"].to(get_device())
    feature_motion = dataset[test_id_idx]["motion"].to(get_device())
    feature_emotion = dataset[test_id_idx]["emotion"].to(get_device())

    feature_scene_offset = feature_scene_offset.unsqueeze(0)
    feature_motion = feature_motion.unsqueeze(0)
    feature_emotion = feature_emotion.unsqueeze(0)

    if args.is_video:
        vispath = VIS_MODELS_PATH
    else:
        vispath = "no_video"
                
    os.makedirs(os.path.join(args.output_dir, str(args.test_id)), exist_ok=True)
    print("Using primer index:", test_id_idx, "(", dataset.data_files_chord[test_id_idx], ")")

    if "major" in custumKey:
        feature_key = torch.tensor([0])
        feature_key = feature_key.float()
    elif "minor" in custumKey:
        feature_key = torch.tensor([1])
        feature_key = feature_key.float()
    else:
        feature_key = dataset[test_id_idx]["key"]
    feature_key = feature_key.to(get_device())

    if args.is_video:
        model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                    max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, rpr=args.rpr).to(get_device())
        
        model.load_state_dict(torch.load(args.model_weights))
        modelReg = VideoRegression(max_sequence_video=args.max_sequence_video, total_vf_dim=total_vf_dim, regModel= regModel).to(get_device())
        modelReg.load_state_dict(torch.load(args.modelReg_weights))
    else:
        model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_chord=args.max_sequence_chord, rpr=args.rpr).to(get_device())

    if not isPrimer:
        primerCID = []
        primerCID_root = []
        primerCID_attr = []

        args.num_prime_chord = 1
        if int( feature_key.item() ) == 0:
            primer_user = "C"
        else:
            primer_user = "A:min"
        
        chordID = chordDic[primer_user]
        primerCID.append(chordID)

        chord_arr = primer_user.split(":")
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
        primerCID = primerCID.to(get_device())

        primerCID_root = np.array(primerCID_root)
        primerCID_root = torch.from_numpy(primerCID_root)
        primerCID_root = primerCID_root.to(torch.long)
        primerCID_root = primerCID_root.to(get_device())
        
        primerCID_attr = np.array(primerCID_attr)
        primerCID_attr = torch.from_numpy(primerCID_attr)
        primerCID_attr = primerCID_attr.to(torch.long)
        primerCID_attr = primerCID_attr.to(get_device())
    else:
        if len(custumPrimer) >= 1:
            primerCID = []
            primerCID_root = []
            primerCID_attr = []
            
            for pChord in custumPrimer:
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
            primerCID = primerCID.to(get_device())

            primerCID_root = np.array(primerCID_root)
            primerCID_root = torch.from_numpy(primerCID_root)
            primerCID_root = primerCID_root.to(torch.long)
            primerCID_root = primerCID_root.to(get_device())
            
            primerCID_attr = np.array(primerCID_attr)
            primerCID_attr = torch.from_numpy(primerCID_attr)
            primerCID_attr = primerCID_attr.to(torch.long)
            primerCID_attr = primerCID_attr.to(get_device())

    # GENERATION
    model.eval()
    with torch.set_grad_enabled(False):
        if(args.beam > 0):
            print("BEAM:", args.beam)
            assert False, "No Beam sampling method implemented yet..."
        else:
            print("RAND DIST")

            if custumKey != "":
                f_path_midi = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + custumKey + "_cgen_rd.mid")
                f_path_lab = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + custumKey + "_cgen_rd.lab")
                f_path_flac = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + custumKey + "_cgen_rd.flac")
                f_path_video = "dataset/vevo/" + str(args.test_id) +".mp4"
                f_path_video_out = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + custumKey + "_cgen_rd.mp4")    
            else:
                f_path_midi = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + "_cgen_rd.mid")
                f_path_lab = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + "_cgen_rd.lab")
                f_path_flac = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + "_cgen_rd.flac")
                f_path_video = "dataset/vevo/" + str(args.test_id) +".mp4"
                f_path_video_out = os.path.join(args.output_dir, str(args.test_id), str(args.test_id) + "_cgen_rd.mp4")

            if args.is_video:
                if isPrimer and len(custumPrimer) == 0:
                    rand_seq = model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primer[:args.num_prime_chord],
                                              primer_root = primer_root[:args.num_prime_chord],
                                              primer_attr = primer_attr[:args.num_prime_chord],
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord)
                else:
                    rand_seq = model.generate(feature_semantic_list=feature_semantic_list, 
                                              feature_key=feature_key, 
                                              feature_scene_offset=feature_scene_offset,
                                              feature_motion=feature_motion,
                                              feature_emotion=feature_emotion,
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0,
                                              max_conseq_N= max_conseq_N,
                                              max_conseq_chord = max_conseq_chord)
                vispath = VIS_MODELS_PATH
                modelReg.eval()
                with torch.set_grad_enabled(False):
                    y = modelReg(
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
                    if density <= 5:
                        densitylist.append(0)
                    elif density <= 10:
                        densitylist.append(1)
                    elif density <= 15:
                        densitylist.append(2)
                    elif density <= 20:
                        densitylist.append(3)
                    else:
                        densitylist.append(4)

                # generated ChordID to ChordSymbol
                chord_genlist = []
                chordID_genlist= rand_seq[0].cpu().numpy()
                for i in chordID_genlist:
                    chord_genlist.append(chordInvDic[str(i)])
                
                chord_offsetlist = convert_format_id_to_offset(chord_genlist)
                
                # Write lab file
                with open(f_path_lab,'w',encoding = 'utf-8') as f:
                    f.write("key ?"+"\n")
                    for i in range(0, len(chord_genlist)):
                        f.write(str(i) + " "+chord_genlist[i]+"\n")
                
                # ChordSymbol to MIDI file with voicing
                MIDI = MIDIFile(1)
                MIDI.addTempo(0, 0, tempo)

                midi_chords_orginal = []
                for i, key in enumerate(chord_genlist):
                    key = key.replace(":", "")
                    if key == "N":
                        midi_chords_orginal.append([])
                    else:
                        midi_chords_orginal.append(Chord(key).getMIDI("c", 4))
                if is_voice:
                    midi_chords = voice(midi_chords_orginal)
                else:
                    midi_chords = midi_chords_orginal
                
                if isArp:
                    #chord_genlist
                    for i, chord in enumerate(midi_chords):
                        if densitylist[i] == 0:
                            # 1 * * * 2 * * * | 3 * * * 4 * * *  
                            # 1 * * * 3 * * * | 2 * * * 3 * * * 
                            if len(chord) == 4:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                            elif len(chord) == 5:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                        elif densitylist[i] == 1:
                            # 1 * 2 * 3 * * * | 4 * 2 * 3 * * *
                            # 1 * 3 * 2 * * * | 4 * 3 * 2 * * *
                            if len(chord) == 4:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                            elif len(chord) == 5:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                        elif densitylist[i] == 2:
                            # 1 * 2 * 3 * 4 * | 3 * 2 * 3 * 4 *
                            # 1 * 3 * 2 * 3 * | 4 * 3 * 2 * 3 *
                            if len(chord) == 4:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velolistExp[i])
                            elif len(chord) == 5:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velolistExp[i])
                        elif densitylist[i] == 3:
                            # 1 2 3 2 4 * 3 * | 2 1 2 3 4 * 3 * 
                            # 1 2 3 4 3 * 4 * | 2 1 2 3 4 * 3 * 
                            if len(chord) == 4:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.5 ,  duration,  velolistExp[i])
                            elif len(chord) == 5:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.5 ,  duration,  velolistExp[i])
                        elif densitylist[i] == 4:
                            # 1 2 3 2 4 3 2 3 | 2 1 2 3 4 3 2 3 
                            # 1 2 3 4 3 2 3 4 | 2 1 2 3 4 2 3 4
                            if len(chord) == 4:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.75 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.75 ,  duration,  velolistExp[i])
                            elif len(chord) == 5:
                                if chord_offsetlist[i] % 2 == 0:
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.75 ,  duration,  velolistExp[i])
                                else:
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[0],  i * duration + 0.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 0.75 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[3],  i * duration + 1 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.25 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[1],  i * duration + 1.5 ,  duration,  velolistExp[i])
                                    MIDI.addNote(0, 0, chord[2],  i * duration + 1.75 ,  duration,  velolistExp[i])
                else:
                    for i, chord in enumerate(midi_chords):
                        for pitch in chord:
                            MIDI.addNote(0, 0, pitch,  i * duration,  duration,  velolistExp[i])
            else:
                if isPrimer and len(custumPrimer) == 0:
                    rand_seq = model.generate(feature_key=feature_key, 
                                              primer = primer[:args.num_prime_chord],
                                              primer_root = primer_root[:args.num_prime_chord],
                                              primer_attr = primer_attr[:args.num_prime_chord],
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0)
                else:
                    rand_seq = model.generate(feature_key=feature_key, 
                                              primer = primerCID, 
                                              primer_root = primerCID_root,
                                              primer_attr = primerCID_attr,
                                              target_seq_length = args.target_seq_length_chord, 
                                              beam=0)
                vispath = "no_video"
                
                chord_genlist = []
                chordID_genlist= rand_seq[0].cpu().numpy()
                for i in chordID_genlist:
                    chord_genlist.append(chordInvDic[str(i)])
                
                chord_offsetlist = convert_format_id_to_offset(chord_genlist)
                
                # Write lab file
                with open(f_path_lab,'w',encoding = 'utf-8') as f:
                    f.write("key ?"+"\n")
                    for i in range(0, len(chord_genlist)):
                        f.write(str(i) + " "+chord_genlist[i]+"\n")
                
                # ChordSymbol to MIDI file with voicing
                MIDI = MIDIFile(1)
                MIDI.addTempo(0, 0, tempo)

                midi_chords_orginal = []
                for i, key in enumerate(chord_genlist):
                    key = key.replace(":", "")
                    if key == "N":
                        midi_chords_orginal.append([])
                    else:
                        midi_chords_orginal.append(Chord(key).getMIDI("c", 4))
                if is_voice:
                    midi_chords = voice(midi_chords_orginal)
                else:
                    midi_chords = midi_chords_orginal
                
                if isArp:
                    #chord_genlist
                    for i, chord in enumerate(midi_chords):
                        # 1 * 2 * 3 * 4 * | 3 * 2 * 3 * 4 *
                        # 1 * 3 * 2 * 3 * | 4 * 3 * 2 * 3 *
                        if len(chord) == 4:
                            if chord_offsetlist[i] % 2 == 0:
                                MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velocity)
                            else:
                                MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velocity)
                        elif len(chord) == 5:
                            if chord_offsetlist[i] % 2 == 0:
                                MIDI.addNote(0, 0, chord[0],  i * duration + 0 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velocity)
                            else:
                                MIDI.addNote(0, 0, chord[2],  i * duration + 0 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[1],  i * duration + 0.5 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[2],  i * duration + 1 ,  duration,  velocity)
                                MIDI.addNote(0, 0, chord[3],  i * duration + 1.5 ,  duration,  velocity)
                else:
                    for i, chord in enumerate(midi_chords):
                        for pitch in chord:
                            MIDI.addNote(0, 0, pitch,  i * duration,  duration,  velocity)

            # Write midi file
            with open(f_path_midi, "wb") as outputFile:
                MIDI.writeFile(outputFile)
            
            # Convert midi to audio (e.g., flac)
            fs = FluidSynth()
            fs.midi_to_audio(f_path_midi, f_path_flac)

            # Render generated music into input video
            audio=mp.AudioFileClip(f_path_flac)
            video=mp.VideoFileClip(f_path_video)
            audio = audio.subclip(0, video.duration )
            final=video.set_audio(audio)
            
            text_prime = text_clip("Prime Chords", args.num_prime_chord)
            text_gen = text_clip("Generated Chords", int(video.duration) - args.num_prime_chord, args.num_prime_chord)

            final_with_text = CompositeVideoClip([final, text_prime, text_gen])
            final_with_text.write_videofile(f_path_video_out, 
                codec='libx264', 
                audio_codec='aac', 
                temp_audiofile='temp-audio.m4a', 
                remove_temp=True
            )

if __name__ == "__main__":
    main()
    
