import os
import wandb
import torch

from tqdm import tqdm
from src.utils.average_meter import AverageMeter


def get_optimizer(optimizer_type, list_model_params, learning_rate=1e-4):
    # Same weight_decay for encoder and predictor - could be changed to an individual weight decay
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            list_model_params,
            lr=learning_rate,
            weight_decay=1e-4,
        )
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            list_model_params,
            lr=learning_rate,
            weight_decay=1e-4,
        )
    else:
        assert False, f"Optimizer type {optimizer_type} not supported."
    
    return optimizer


# ----------------------
# PHASE 1: PRETRAINING
# ----------------------
def train_jepa_framework(model, dataloader, evaluator, loss_fn, training_config={}):

    optimizer_type     = training_config['optimizer_type']
    output_dir         = training_config['output_dir']
    epochs             = training_config['epochs']
    epochs_to_evaluate = training_config['epochs_to_evaluate']
    pbar   = tqdm(total=int(epochs * len(dataloader.dataset) / dataloader.batch_size), desc="init training...".center(50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = get_optimizer(optimizer_type    = optimizer_type,
                              list_model_params = list(model.encoder_online.parameters()) + list(model.predictor.parameters()),
                              learning_rate     = training_config['learning_rate'])

        
    model.train()
    for epoch in range(epochs):
        loss_record = AverageMeter()
        batch_pass  = 0
        total_loss  = 0.0
        avg_loss    = 0.0

        # Evaluate the model before training
        if epoch in epochs_to_evaluate:
            model.eval()
            evaluator.evaluate(model, epoch=epoch, training_phase='pretrain')
            model.train()


        for batch in dataloader:  
            labels = batch['objects']['label']
            if len(labels) == 0:
                # No objects remain in this batch from the preprocessing step
                continue          
            x_whole = batch['objects']['whole'].to(device)

            if model.data_source == 'automotive_scene_level':
                # Scene-based dimensions: BatchSize(=1) x Samples x TimeSteps x Features --> [BatchSize x TimeSteps x Features]
                if len(x_whole.shape) == 4:
                    # Remove the sequence dimension if it exists
                    x_whole   = x_whole.squeeze()
                if len(x_whole.shape) < 3:
                    # Add additional dimension
                    x_whole   = x_whole.unsqueeze(dim=0)  
            
            # Exectution of model / forward pass
            z_dict = model.forward_train(x_whole)
            
            # Loss
            loss = loss_fn(y_pred=z_dict['z_ctx_masked'], y=z_dict['z_tgt_masked'])

            # Backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
                        
            # ensure optimizer param list is only online/predictor params
            opt_params = set([id(p) for g in optimizer.param_groups for p in g['params']])
            for p in model.encoder_target.parameters():
               assert id(p) not in opt_params, "Target encoder parameters must not be in optimizer"    
                
            batch_pass += 1
            total_loss += loss.item()

            # Update EMA target encoder
            model.update_target_ema()

            # === Progress bar ===
            pbar.update(1)
            loss_record.update(loss.item(), loss)
            log_msg = "Epoch:{:2}/{}  Iter:{:3}/{} Avg Loss: {:6.3f}".format(
                            epoch + 1, epochs, 
                            batch_pass, len(dataloader),
                            round(loss_record.avg.item(), 3)).center(50)
            pbar.set_description(log_msg)
            
        avg_loss = total_loss / len(dataloader)
        # Each epoch log to wandb
        log_dict = {'Epoch':    epoch+1,
                    'Avg Loss': avg_loss}
        
        wandb.log(log_dict)    


    # Save pretraining weights
    path_pretrain_weights = os.path.join(output_dir, "pretrain_weights.pth")
    torch.save(model.state_dict(), path_pretrain_weights)

    # Final Evaluation of the model
    model.eval()
    evaluator.evaluate(model, epoch=epoch+1)
    model.train()

    print("Pretraining complete")

