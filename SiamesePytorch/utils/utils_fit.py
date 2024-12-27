import os

import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import get_lr


def fit_one_epoch(model_train, model, loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, genval, Epoch, cuda, fp16, scaler, save_period, save_dir, loss_weights, local_rank=0):
    total_loss      = 0
    total_accuracy  = 0

    val_loss            = 0
    val_total_accuracy  = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        img_dst, img_ref, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                img_dst  = img_dst.cuda(local_rank)
                img_ref  = img_ref.cuda(local_rank)
                targets = targets.cuda(local_rank)
                lw = torch.tensor([loss_weights[str(round(x.item(), 4))] for x in targets]).cuda(local_rank)

        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(img_dst, img_ref).squeeze(-1)
            output  = loss(outputs, targets) * lw
            output = output.mean()
            output.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(img_dst, img_ref).squeeze(-1)
                output  = loss(outputs, targets) * lw
                output = output.mean()
            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(output).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        total_loss      += output.item()
        total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'acc'       : total_accuracy / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_step_val:
            break
        
        img_dst, img_ref, targets = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                img_dst  = img_dst.cuda(local_rank)
                img_ref  = img_ref.cuda(local_rank)
                targets = targets.cuda(local_rank)
                lw = torch.tensor([loss_weights[str(round(x.item(), 4))] for x in targets]).cuda(local_rank)
                
            optimizer.zero_grad()
            outputs = model_train(img_dst, img_ref).squeeze(-1)
            output  = loss(outputs, targets) * lw
            output  = output.mean()

            equal       = torch.eq(torch.round(nn.Sigmoid()(outputs)), targets)
            accuracy    = torch.mean(equal.float())

        val_loss            += output.item()
        val_total_accuracy  += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1), 
                                'acc'       : val_total_accuracy / (iteration + 1)})
            pbar.update(1)
                
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(state, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(state, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(state, os.path.join(save_dir, "last_epoch_weights.pth"))