
import os
import cv2
import torch
import shutil
import numpy as np
from tabulate import tabulate
from torch.utils.data import DataLoader

def train(model,
          train_dataset,
          val_dataset=None,
          optimizer=None,
          lr_scheduler=None,
          save_dir='runs', 
          iters=10000,
          batch_size=2,
          save_interval=1000,
          logger=None,
          log_iters=10,
          num_workers=0,
          criterion=None,
          device=None,
          class_names=[],
          visualizer=None,
          keep_checkpoint_max=5):
    model_dir = os.path.join(save_dir,'model')
    result_dir = os.path.join(save_dir,'results')
    os.makedirs(model_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)

    model.train()
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(train_dataset,batch_size=1, shuffle=False, num_workers=num_workers)

    start_iter = 0
    best_mean_iou = -1.0
    best_model_iter = -1
    avg_loss = 0.0
    iter = start_iter
    iters_per_epoch = len(train_dataset)//batch_size
    model_paths = []
    torch.optim.lr_scheduler.MultiStepLR.get_lr
    print("-------- TRAINING  BEGINING ------------")
    while iter < iters:
        for data in train_loader:
            iter += 1
            if iter > iters:
                break
            images = data['img'].to(device)
            labels = data['label'].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = sum([criterion(logit,labels) for logit in logits])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            avg_loss += loss.item()
            if (iter) % log_iters == 0:
                avg_loss /= log_iters
                lr = lr_scheduler.get_last_lr()[0]
                logger.info("[TRAIN] epoch: {}, iter: {}/{}, loss: {:.4f}, lr: {:.6f}".format((iter - 1) // iters_per_epoch + 1, iter, iters, avg_loss,lr))
                avg_loss = 0.0
            if (iter % save_interval == 0 or iter == iters) and (val_dataset is not None):
                mean_iou, aAcc = evaluate(model,val_dataset,class_names,device)
                model_path = os.path.join(save_dir, f'iter_{iter}.pth')
                model_paths.append(model_path)
                torch.save(model.state_dict(),model_path)
                if len(model_paths) > keep_checkpoint_max > 0:
                    os.remove(model_paths.pop(0))
                if val_dataset is not None:
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        best_model_iter = iter
                        best_model_path = os.path.join(save_dir, f'best.pth')
                        torch.save(model.state_dict(), best_model_path)
                        logger.info('[EVAL] The model with the best validation mIoU ({:.4f}) was saved at iter {}.'.format(best_mean_iou, best_model_iter))
    print("========== TRAINING FINISHED ===========")

def evaluate(model,dataset,class_names=[],device=None,num_workers=1,metrics=['aAcc','IoU','Acc','Precision','Dice']):
    model.eval()
    loader = DataLoader(dataset,batch_size=1, shuffle=False, num_workers=num_workers)
    num_classes = len(class_names)
    results = []
    with torch.no_grad():
        for iter, data in enumerate(loader):
            images = data['img'].to(device)
            label = data['label'].to(device).squeeze()
            logit = model(images)[0]
            mask = torch.argmax(logit,dim=1)[0]
            results.append(intersect_and_union(label,mask,num_classes))
    
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    headers = []
    datas = []
    ret_metrics = []
    mean_iou = 0
    all_acc = 0
    for metric in metrics:
        if metric == 'IoU':
            headers.append('mIoU')
            headers.append('IoU')
            iou = total_area_intersect / (total_area_union+1)
            mean_iou = iou[1:].mean()
            datas.append([mean_iou]*num_classes)
            datas.append(list(iou))
        elif metric == 'aAcc':
            headers.append('aAcc')
            all_acc = total_area_intersect[1:].sum() / total_area_label[1:].sum()
            datas.append([all_acc]*num_classes)
        elif metric == 'Acc':
            headers.append('Acc')
            acc = total_area_intersect / (total_area_label+1)
            datas.append(list(acc))
        elif metric == 'Precision':
            headers.append('Precision')
            precision = total_area_intersect / total_area_pred_label
            datas.append(list(precision))
        elif metric == 'Dice':
            headers.append('Dice')
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label+1)
            datas.append(list(dice))

    for i,line in enumerate(zip(*datas)):
        ret_metrics.append([class_names[i],]+list(line))  

    print(tabulate(ret_metrics,headers))  

    return mean_iou, all_acc


def intersect_and_union(pred_label, label, num_classes, ignore_index=255):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(intersect.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_pred_label = torch.histc(pred_label.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_label = torch.histc(label.float(), bins=(num_classes), min=0,max=num_classes - 1).cpu()
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label

