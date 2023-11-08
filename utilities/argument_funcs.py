import argparse
from .constants import *

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_dir", type=str, default="./dataset/", help="Folder of VEVO dataset")
    
    parser.add_argument("-input_dir_music", type=str, default="./dataset/vevo_chord/" + MUSIC_TYPE, help="Folder of video CNN feature files")
    parser.add_argument("-input_dir_video", type=str, default="./dataset/vevo_vis", help="Folder of video CNN feature files")

    parser.add_argument("-output_dir", type=str, default="./saved_models", help="Folder to save model weights. Saves one every epoch")
    
    parser.add_argument("-weight_modulus", type=int, default=1, help="How often to save epoch weights (ex: value of 10 means save every 10 epochs)")
    parser.add_argument("-print_modulus", type=int, default=1, help="How often to print train results for a batch (batch loss, learn rate, etc.)")
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")
    parser.add_argument("--no_tensorboard", action="store_true", help="Turns off tensorboard result reporting")
    parser.add_argument("-continue_weights", type=str, default=None, help="Model weights to continue training based on")
    parser.add_argument("-continue_epoch", type=int, default=None, help="Epoch the continue_weights model was at")
    parser.add_argument("-lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("-ce_smoothing", type=float, default=None, help="Smoothing parameter for smoothed cross entropy loss (defaults to no smoothing)")
    parser.add_argument("-batch_size", type=int, default=1, help="Batch size to use")
    parser.add_argument("-epochs", type=int, default=5, help="Number of epochs to use")

    parser.add_argument("-max_sequence_midi", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-max_sequence_video", type=int, default=300, help="Maximum video sequence to consider")
    parser.add_argument("-max_sequence_chord", type=int, default=300, help="Maximum video sequence to consider")

    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument("-dropout", type=float, default=0.1, help="Dropout rate")

    parser.add_argument("-is_video", type=bool, default=IS_VIDEO, help="MusicTransformer or VideoMusicTransformer")

    if IS_VIDEO:
        parser.add_argument("-vis_models", type=str, default=VIS_MODELS_SORTED, help="...")
    else:
        parser.add_argument("-vis_models", type=str, default="", help="...")

    parser.add_argument("-emo_model", type=str, default="6c_l14p", help="...")
    parser.add_argument("-rpr", type=bool, default=RPR, help="...")
    return parser.parse_args()

def print_train_args(args):
    print(SEPERATOR)
    
    print("dataset_dir:", args.dataset_dir )
    
    print("input_dir_music:", args.input_dir_music)
    print("input_dir_video:", args.input_dir_video)

    print("output_dir:", args.output_dir)

    print("weight_modulus:", args.weight_modulus)
    print("print_modulus:", args.print_modulus)
    print("")
    print("n_workers:", args.n_workers)
    print("force_cpu:", args.force_cpu)
    print("tensorboard:", not args.no_tensorboard)
    print("")
    print("continue_weights:", args.continue_weights)
    print("continue_epoch:", args.continue_epoch)
    print("")
    print("lr:", args.lr)
    print("ce_smoothing:", args.ce_smoothing)
    print("batch_size:", args.batch_size)
    print("epochs:", args.epochs)
    print("")
    print("rpr:", args.rpr)

    print("max_sequence_midi:", args.max_sequence_midi)
    print("max_sequence_video:", args.max_sequence_video)
    print("max_sequence_chord:", args.max_sequence_chord)
    
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("")
    print("dim_feedforward:", args.dim_feedforward)
    print("dropout:", args.dropout)
    print("is_video:", args.is_video)

    print(SEPERATOR)
    print("")

def parse_eval_args():
    if IS_VIDEO:
        modelpath = "./saved_models/AMT/best_acc_weights.pickle"
        # modelpath = "./saved_models/"+version+ "/"+VIS_MODELS_PATH+"/results/best_loss_weights.pickle"
    else:
        modelpath = "./saved_models/"+version+ "/no_video/results/best_acc_weights.pickle"

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset_dir", type=str, default="./dataset/", help="Folder of VEVO dataset")
    
    parser.add_argument("-input_dir_music", type=str, default="./dataset/vevo_chord/" + MUSIC_TYPE, help="Folder of video CNN feature files")
    parser.add_argument("-input_dir_video", type=str, default="./dataset/vevo_vis", help="Folder of video CNN feature files")
    
    parser.add_argument("-model_weights", type=str, default= modelpath, help="Pickled model weights file saved with torch.save and model.state_dict()")
    
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")
    parser.add_argument("-batch_size", type=int, default=1, help="Batch size to use")
    
    parser.add_argument("-max_sequence_midi", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-max_sequence_video", type=int, default=300, help="Maximum video sequence to consider")
    parser.add_argument("-max_sequence_chord", type=int, default=300, help="Maximum video sequence to consider")

    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

    parser.add_argument("-is_video", type=bool, default=IS_VIDEO, help="MusicTransformer or VideoMusicTransformer")

    if IS_VIDEO:
        parser.add_argument("-vis_models", type=str, default=VIS_MODELS_SORTED, help="...")
    else:
        parser.add_argument("-vis_models", type=str, default="", help="...")

    parser.add_argument("-emo_model", type=str, default="6c_l14p", help="...")
    parser.add_argument("-rpr", type=bool, default=RPR, help="...")
    return parser.parse_args()

def print_eval_args(args):
    print(SEPERATOR)
    print("input_dir_music:", args.input_dir_music)
    print("input_dir_video:", args.input_dir_video)

    print("model_weights:", args.model_weights)
    print("n_workers:", args.n_workers)
    print("force_cpu:", args.force_cpu)
    print("")
    print("batch_size:", args.batch_size)
    print("")
    print("rpr:", args.rpr)
    
    print("max_sequence_midi:", args.max_sequence_midi)
    print("max_sequence_video:", args.max_sequence_video)
    print("max_sequence_chord:", args.max_sequence_chord)
    
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("")
    print("dim_feedforward:", args.dim_feedforward)
    print(SEPERATOR)
    print("")

# parse_generate_args
def parse_generate_args():
    parser = argparse.ArgumentParser()
    outputpath = "./output_vevo/"+version
    if IS_VIDEO:
        modelpath = "./saved_models/AMT/best_loss_weights.pickle"
        modelpathReg = "./saved_models/AMT/best_rmse_weights.pickle"
        # modelpath = "./saved_models/"+version+ "/"+VIS_MODELS_PATH+"/results/best_acc_weights.pickle"
        # modelpathReg = "./saved_models/"+version+ "/"+VIS_MODELS_PATH+"/results_regression_bigru/best_rmse_weights.pickle"
    else:
        modelpath = "./saved_models/"+version+ "/no_video/results/best_loss_weights.pickle"
        modelpathReg = None

    parser.add_argument("-dataset_dir", type=str, default="./dataset/", help="Folder of VEVO dataset")
    
    parser.add_argument("-input_dir_music", type=str, default="./dataset/vevo_chord/" + MUSIC_TYPE, help="Folder of video CNN feature files")
    parser.add_argument("-input_dir_video", type=str, default="./dataset/vevo_vis", help="Folder of video CNN feature files")

    parser.add_argument("-output_dir", type=str, default= outputpath, help="Folder to write generated midi to")

    parser.add_argument("-primer_file", type=str, default=None, help="File path or integer index to the evaluation dataset. Default is to select a random index.")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

    parser.add_argument("-target_seq_length_midi", type=int, default=1024, help="Target length you'd like the midi to be")
    parser.add_argument("-target_seq_length_chord", type=int, default=300, help="Target length you'd like the midi to be")
    
    parser.add_argument("-num_prime_midi", type=int, default=256, help="Amount of messages to prime the generator with")
    parser.add_argument("-num_prime_chord", type=int, default=30, help="Amount of messages to prime the generator with")    
    parser.add_argument("-model_weights", type=str, default=modelpath, help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-modelReg_weights", type=str, default=modelpathReg, help="Pickled model weights file saved with torch.save and model.state_dict()")

    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")

    parser.add_argument("-max_sequence_midi", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-max_sequence_video", type=int, default=300, help="Maximum video sequence to consider")
    parser.add_argument("-max_sequence_chord", type=int, default=300, help="Maximum chord sequence to consider")

    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")

    parser.add_argument("-is_video", type=bool, default=IS_VIDEO, help="MusicTransformer or VideoMusicTransformer")

    if IS_VIDEO:
        parser.add_argument("-vis_models", type=str, default=VIS_MODELS_SORTED, help="...")
    else:
        parser.add_argument("-vis_models", type=str, default="", help="...")

    parser.add_argument("-emo_model", type=str, default="6c_l14p", help="...")
    parser.add_argument("-rpr", type=bool, default=RPR, help="...")
    parser.add_argument("-test_id", type=str, default=None, help="Dimension of the feedforward layer")

    return parser.parse_args()

def print_generate_args(args):
    
    print(SEPERATOR)
    print("input_dir_music:", args.input_dir_music)
    print("input_dir_video:", args.input_dir_video)

    print("output_dir:", args.output_dir)
    print("primer_file:", args.primer_file)
    print("force_cpu:", args.force_cpu)
    print("")

    print("target_seq_length_midi:", args.target_seq_length_midi)
    print("target_seq_length_chord:", args.target_seq_length_chord)
    
    print("num_prime_midi:", args.num_prime_midi)
    print("num_prime_chord:", args.num_prime_chord)

    print("model_weights:", args.model_weights)
    print("beam:", args.beam)
    print("")
    print("rpr:", args.rpr)
    
    print("max_sequence_midi:", args.max_sequence_midi)
    print("max_sequence_video:", args.max_sequence_video)
    print("max_sequence_chord:", args.max_sequence_chord)
    

    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("")
    print("dim_feedforward:", args.dim_feedforward)
    print("")
    print("test_id:", args.test_id)

    print(SEPERATOR)
    print("")

# write_model_params
def write_model_params(args, output_file):
    o_stream = open(output_file, "w")

    o_stream.write("rpr: " + str(args.rpr) + "\n")
    o_stream.write("lr: " + str(args.lr) + "\n")
    o_stream.write("ce_smoothing: " + str(args.ce_smoothing) + "\n")
    o_stream.write("batch_size: " + str(args.batch_size) + "\n")

    o_stream.write("max_sequence_midi: " + str(args.max_sequence_midi) + "\n")
    o_stream.write("max_sequence_video: " + str(args.max_sequence_video) + "\n")
    o_stream.write("max_sequence_chord: " + str(args.max_sequence_chord) + "\n")
    
    o_stream.write("n_layers: " + str(args.n_layers) + "\n")
    o_stream.write("num_heads: " + str(args.num_heads) + "\n")
    o_stream.write("d_model: " + str(args.d_model) + "\n")
    o_stream.write("dim_feedforward: " + str(args.dim_feedforward) + "\n")
    o_stream.write("dropout: " + str(args.dropout) + "\n")

    o_stream.write("is_video: " + str(args.is_video) + "\n")
    o_stream.write("vis_models: " + str(args.vis_models) + "\n")
    o_stream.write("input_dir_music: " + str(args.input_dir_music) + "\n")
    o_stream.write("input_dir_video: " + str(args.input_dir_video) + "\n")

    o_stream.close()
