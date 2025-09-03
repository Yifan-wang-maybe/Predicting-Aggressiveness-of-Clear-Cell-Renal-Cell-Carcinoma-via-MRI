# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from dice import DiceLoss
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score,precision_recall_curve, auc

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    torch.cuda.empty_cache()
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    OUTPUT = []
    LABEL = []
    
    #criterion_score = torch.nn.CrossEntropyLoss()
    pos_weight = torch.tensor([1.0]).to(device)
    #criterion_score = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight = pos_weight)
    criterion_score = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, seg, lesion_label, Mask_location) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        Mask_location = Mask_location.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        lesion_label = lesion_label.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, seg, lesion_label, Mask_location = mixup_fn(samples, seg, lesion_label, Mask_location)
        
        with torch.cuda.amp.autocast():
            output = model(samples,Mask_location)   ### [BS, 678, 15, 15]
            loss_score = criterion_score(output[:,0,0].float(),lesion_label.float())
            loss = loss_score


            OUTPUT.append(output[:,0,0].detach().cpu().numpy())
            LABEL.append(lesion_label.detach().cpu().numpy())


        loss_value = loss.item()
                
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)


        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
        
        #torch.cuda.empty_cache()

    OUTPUT = [item for sublist in OUTPUT for item in sublist]
    LABEL = [item for sublist in LABEL for item in sublist]
    
    OUTPUT = np.array(OUTPUT)
    LABEL = np.array(LABEL)
    
    print('Training_AUC:',get_AUC(OUTPUT,LABEL))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def get_AUC(outputs, labels):
    
    labels = labels.reshape(-1)
    outputs = outputs.reshape(-1)

    if sum(labels) == 0:
        labels[0] = 1  
    
    if sum(labels) == len(labels):
        labels[0] = 0  
    
    precision, recall, thresholds = precision_recall_curve(labels, outputs)
    
    pr_auc = auc(recall, precision)

    return pr_auc 

    

@torch.no_grad()
def evaluate(data_loader, model, device, test, max_IOU, save):
    torch.cuda.empty_cache()
    #criterion = torch.nn.BCEWithLogitsLoss()
    pos_weight = torch.tensor([1.0]).to(device)
    #criterion_score = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight = pos_weight)
    criterion_score = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval:'

    # switch to evaluation mode
    model.eval()
    OUTPUT = []
    LABEL = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        seg = batch[1]
        lesion_label = batch[2]
        Mask_location = batch[3]

        images = images.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)
        lesion_label = lesion_label.to(device, non_blocking=True)
        Mask_location = Mask_location.to(device, non_blocking=True)    

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images,Mask_location)
            loss_score = criterion_score(output[:,0,0].float(),lesion_label.float())
            loss = loss_score

            output_sig = F.logsigmoid(output[:,0,0]).exp()

            OUTPUT.append(output_sig.detach().cpu().numpy())
            LABEL.append(lesion_label.detach().cpu().numpy())
            

        #acc1, acc5 = accuracy(output, target, topk=(1, 5))    
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())


    # gather the stats from all processes
   
    metric_logger.synchronize_between_processes()
    #print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #      .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    #print('wwwww',metric_logger.loss)
    
    
    #print('Eval: loss {losses.global_avg:.3f} IOU_0 {IOU_0.global_avg:.3f} IOU_1 {IOU_1.global_avg:.3f} IOU_2 {IOU_2.global_avg:.3f} IOU_3 {IOU_3.global_avg:.3f} IOU_4 {IOU_4.global_avg:.3f}'
    #      .format(losses=metric_logger.loss, IOU_0 = metric_logger.IOU_0, IOU_1 = metric_logger.IOU_1,IOU_2 = metric_logger.IOU_2,IOU_3 = metric_logger.IOU_3,IOU_4 = metric_logger.IOU_4))
    
    #print('Eval: loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    
    OUTPUT = [item for sublist in OUTPUT for item in sublist]
    LABEL = [item for sublist in LABEL for item in sublist]
    
    
    OUTPUT = np.array(OUTPUT)
    LABEL = np.array(LABEL)

    if test == 0:
        if max_IOU > metric_logger.meters['loss'].avg:
            print(OUTPUT)
            print(LABEL)


    if save == 1:
        if test == 0:
            np.save('Cross_OUTPUT_1.npy',OUTPUT)
            np.save('Cross_LABEL_1.npy',LABEL)
        else:       
            np.save('OUTPUT_1.npy',OUTPUT)
            np.save('LABEL_1.npy',LABEL)

    if test == 0:
        print('Val_AUC:',get_AUC(OUTPUT,LABEL))
    else:
        print('Test_AUC:',get_AUC(OUTPUT,LABEL))
    

    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger.meters['loss'].avg



