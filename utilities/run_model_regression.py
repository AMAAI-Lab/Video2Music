import torch
import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr
import torch.nn.functional as F

def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):
    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()
        opt.zero_grad()

        feature_semantic_list = [] 
        for feature_semantic in batch["semanticList"]:
            feature_semantic_list.append( feature_semantic.to(get_device()) )

        feature_scene_offset = batch["scene_offset"].to(get_device())
        feature_motion = batch["motion"].to(get_device())
        feature_emotion = batch["emotion"].to(get_device())

        feature_note_density = batch["note_density"].to(get_device())
        feature_loudness = batch["loudness"].to(get_device())

        y = model(
                  feature_semantic_list, 
                  feature_scene_offset,
                  feature_motion,
                  feature_emotion)
        
        y   = y.reshape(y.shape[0] * y.shape[1], -1)
        
        feature_loudness = feature_loudness.flatten().reshape(-1,1) # (300, 1)
        feature_note_density = feature_note_density.flatten().reshape(-1,1) # (300, 1)        
        feature_combined = torch.cat((feature_note_density, feature_loudness), dim=1) # (300, 2)

        out = loss.forward(y, feature_combined)
        out.backward()
        opt.step()
        
        if(lr_scheduler is not None):
            lr_scheduler.step()
        time_after = time.time()
        time_took = time_after - time_before
        
        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")
    return

def eval_model(model, dataloader, loss):
    model.eval()
    
    avg_rmse     = -1
    avg_loss    = -1
    avg_rmse_note_density     = -1
    avg_rmse_loudness     = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        
        sum_loss   = 0.0
        
        sum_rmse    = 0.0
        sum_rmse_note_density = 0.0
        sum_rmse_loudness = 0.0

        for batch in dataloader:
            feature_semantic_list = [] 
            for feature_semantic in batch["semanticList"]:
                feature_semantic_list.append( feature_semantic.to(get_device()) )

            feature_scene_offset = batch["scene_offset"].to(get_device())
            feature_motion = batch["motion"].to(get_device())
            feature_emotion = batch["emotion"].to(get_device())
            feature_loudness = batch["loudness"].to(get_device())
            feature_note_density = batch["note_density"].to(get_device())
            
            y = model(
                    feature_semantic_list, 
                    feature_scene_offset,
                    feature_motion,
                    feature_emotion)
            
            y   = y.reshape(y.shape[0] * y.shape[1], -1)

            feature_loudness = feature_loudness.flatten().reshape(-1,1) # (300, 1)
            feature_note_density = feature_note_density.flatten().reshape(-1,1) # (300, 1)        
            feature_combined = torch.cat((feature_note_density, feature_loudness), dim=1) # (300, 2)

            mse = F.mse_loss(y, feature_combined)
            rmse = torch.sqrt(mse)
            sum_rmse += float(rmse)

            y_note_density, y_loudness = torch.split(y, split_size_or_sections=1, dim=1)

            mse_note_density = F.mse_loss(y_note_density, feature_note_density)
            rmse_note_density = torch.sqrt(mse_note_density)
            sum_rmse_note_density += float(rmse_note_density)
            
            mse_loudness = F.mse_loss(y_loudness, feature_loudness)
            rmse_loudness = torch.sqrt(mse_loudness)
            sum_rmse_loudness += float(rmse_loudness)

            out = loss.forward(y, feature_combined)
            sum_loss += float(out)
            
        avg_loss    = sum_loss / n_test
        avg_rmse     = sum_rmse / n_test
        avg_rmse_note_density     = sum_rmse_note_density / n_test
        avg_rmse_loudness     = sum_rmse_loudness / n_test

    return avg_loss, avg_rmse, avg_rmse_note_density, avg_rmse_loudness
