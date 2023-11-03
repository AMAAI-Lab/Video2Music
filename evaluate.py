import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.vevo_dataset import create_vevo_datasets


from model.music_transformer import MusicTransformer
from model.video_music_transformer import VideoMusicTransformer

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_funcs import parse_eval_args, print_eval_args
from utilities.run_model_vevo import eval_model
import logging
import os
import sys

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

VIS_MODELS_ARR = [
    "2d/clip_l14p"
]

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler('log/log_eval2.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# main
def main( vm = "", isPrintArgs = True):
    args = parse_eval_args()

    if isPrintArgs:
        print_eval_args(args)

    if vm != "":
        args.vis_models = vm
        
    if args.is_video:
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
        args.model_weights = "./saved_models/" + version + "/best_loss_weights.pickle"
    else:
        vis_abbr_path = "no_video"
        args.model_weights = "./saved_models/" + version + "/best_loss_weights.pickle"
        
    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    
    _, _, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        emo_model = args.emo_model, 
        split_ver = SPLIT_VER, 
        random_seq = True, 
        is_video = args.is_video)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers)

    total_vf_dim = 0
    if args.is_video:
        for vf in test_dataset[0]["semanticList"]:
            total_vf_dim += vf.shape[1]
        total_vf_dim += 1 # Scene_offset
        total_vf_dim += 1 # Motion
        
        # Emotion
        if args.emo_model.startswith("6c"):
            total_vf_dim += 6
        else:
            total_vf_dim += 5
        
    if args.is_video:
        model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, rpr=args.rpr).to(get_device())
    else:
        model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_chord=args.max_sequence_chord, rpr=args.rpr).to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    ##### Not smoothing evaluation loss #####
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    eval_loss_emotion_func = nn.BCEWithLogitsLoss()

    logging.info( f"VIS MODEL: {args.vis_models}" )
    logging.info("Evaluating:")
    model.eval()

    eval_metric_dict = eval_model(model, test_loader, 
                                eval_loss_func, eval_loss_emotion_func,
                                isVideo= args.is_video, isGenConfusionMatrix=True)
        
    eval_total_loss = eval_metric_dict["avg_total_loss"]
    eval_loss_chord = eval_metric_dict["avg_loss_chord"]
    eval_loss_emotion = eval_metric_dict["avg_loss_emotion"]
    eval_h1 = eval_metric_dict["avg_h1"]
    eval_h3 = eval_metric_dict["avg_h3"]
    eval_h5 = eval_metric_dict["avg_h5"]

    logging.info(f"Avg test loss (total): {eval_total_loss:.4f}" )
    logging.info(f"Avg test loss (chord): {eval_loss_chord:.4f}" )
    logging.info(f"Avg test loss (emotion): {eval_loss_emotion:.4f}" )
    logging.info(f"Avg test h1: {eval_h1:.4f}")
    logging.info(f"Avg test h3: {eval_h3:.4f}")
    logging.info(f"Avg test h5: {eval_h5:.4f}")

if __name__ == "__main__":
    if len(VIS_MODELS_ARR) != 0 :
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()


