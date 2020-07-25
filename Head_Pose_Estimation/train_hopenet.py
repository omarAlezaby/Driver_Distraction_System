import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from datasets import *
from hopenet import *
import torch.utils.model_zoo as model_zoo

import psutil

'''
Helper Functions
'''

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


'''
Training
'''

def get_reg_form_cls(pred, prop_func):
    # compute predected prop
    pred_prop = softmax(pred)
    # get regression value for proplities
    idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)
    reg_pred = torch.sum(pred_prop * idx_tensor, dim=1) * 3 - 99 # [-99, 99]
    return reg_pred

def train_epoch(model, prop_func, optimizer, train_loader, logger, epoch_num, 
                cls_criterion, reg_criterion, disp_every):
    
    model.train()
    
    # values to compute
    total_loss = 0
    total_loss_perAng = [0, 0, 0] # yaw, pitch, roll
    total_acc_perAng = [0, 0, 0] # yaw, pitch, roll
    dataset_size = len(train_loader.dataset)
    
    for i, (img, bins_angles, angles, img_name) in enumerate(train_loader):
        
        # load to device 
        img = img.to(device)
        angles = angles.to(device)
        bins_angles = bins_angles.to(device)
        
        # pass image to the model
        pred_yaw, pred_pitch, pred_roll = model(img)
        
        # compute classification losses
        loss_cls_yaw = cls_criterion(pred_yaw, bins_angles[:, 0])
        loss_cls_pitch = cls_criterion(pred_pitch, bins_angles[:, 1])        
        loss_cls_roll = cls_criterion(pred_roll, bins_angles[:, 2])
        
        # get the regression values for the output of the model
        reg_yaw = get_reg_form_cls(pred_yaw, prop_func)
        reg_pitch = get_reg_form_cls(pred_pitch, prop_func)
        reg_roll = get_reg_form_cls(pred_roll, prop_func)
        
        # computer regression losses 
        loss_reg_yaw = reg_criterion(reg_yaw, angles[:, 0])
        loss_reg_pitch = reg_criterion(reg_pitch, angles[:, 1])        
        loss_reg_roll = reg_criterion(reg_roll, angles[:, 2])
        
        # total losses 
        loss_yaw = loss_cls_yaw + alpha * loss_reg_yaw
        loss_pitch = loss_cls_pitch + alpha * loss_reg_pitch        
        loss_roll = loss_cls_roll + alpha * loss_reg_roll
        
        # backprop and optimize 
        loss = loss_yaw + loss_pitch + loss_roll
        optimizer.zero_grad()
        loss.backward()
        # grad clipping, (max_norm=5.0) 
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        # compute total loss
        total_loss += loss.cpu().detach().numpy()
        total_loss_perAng[0] += loss_yaw.cpu().detach().numpy() * batch_size
        total_loss_perAng[1] += loss_pitch.cpu().detach().numpy() * batch_size
        total_loss_perAng[2] += loss_roll.cpu().detach().numpy() * batch_size
        
        # compute Accuracy
        total_acc_perAng[0] += torch.sum(torch.abs(reg_yaw - angles[:, 0])).cpu().detach().numpy()
        total_acc_perAng[1] += torch.sum(torch.abs(reg_pitch - angles[:, 1])).cpu().detach().numpy()    
        total_acc_perAng[2] += torch.sum(torch.abs(reg_roll - angles[:, 2])).cpu().detach().numpy()
        
        # print results and save to logger
        if i % disp_every == 0:
            print(f'Epoch[{epoch_num}], Itr[{i}/{len(train_loader)}], \
            Losses: Yaw {loss_yaw:.2f}, Pitch {loss_pitch:.2f}, Roll {loss_roll:.2f}')

            
    # finish epoch
    total_acc_perAng = [x/dataset_size for x in total_acc_perAng]
    total_loss_perAng = [x/dataset_size for x in total_loss_perAng]
    total_loss /= dataset_size
    print(f'\nEpoch[{epoch_num}] Total Loss: {total_loss}\n\
    Losses Per Angle: Yaw {total_loss_perAng[0]:.2f}, Pitch {total_loss_perAng[1]:.2f}, Roll {total_loss_perAng[2]:.2f}\n\
    Acc Per Angle: Yaw {total_acc_perAng[0]:.2f}, Pitch {total_acc_perAng[1]:.2f}, Roll {total_acc_perAng[2]:.2f}')
    
    # add to logger 
    logger.add_scalar('Total_Train_Losses/total_loss', total_loss, global_step=epoch_num)
    # losses per angle
    logger.add_scalar('Total_Train_Losses/Yaw', total_loss_perAng[0],global_step=epoch_num)
    logger.add_scalar('Total_Train_Losses/Pitch', total_loss_perAng[1],global_step=epoch_num)        
    logger.add_scalar('Total_Train_Losses/Roll', total_loss_perAng[2],global_step=epoch_num)
    #acc per angle
    logger.add_scalar('Total_Train_Acc/Yaw', total_acc_perAng[0],global_step=epoch_num)
    logger.add_scalar('Total_Train_Acc/Pitch', total_acc_perAng[1],global_step=epoch_num)        
    logger.add_scalar('Total_Train_Acc/Roll', total_acc_perAng[2],global_step=epoch_num)

'''
Evaluation
'''
best_loss = 1e6
best_acc = 0


def evaluate(model, prop_func, val_loader, logger, epoch_num, 
                cls_criterion, reg_criterion, disp_every):
    
    print(f'Evaluating Epoch {epoch_num} ....')
    
    model.eval()
    
    # values to compute
    total_loss = 0
    total_loss_perAng = [0, 0, 0] # yaw, pitch, roll
    total_acc_perAng = [0, 0, 0] # yaw, pitch, roll
    dataset_size = len(val_loader.dataset)
    
    for i, (img, bins_angles, angles, img_name) in enumerate(val_loader):
        
        # load to device 
        img = img.to(device)
        angles = angles.to(device)
        bins_angles = bins_angles.to(device)
        
        with torch.no_grad():
            # pass image to the model
            pred_yaw, pred_pitch, pred_roll = model(img)
        
        # compute classification losses
        loss_cls_yaw = cls_criterion(pred_yaw, bins_angles[:, 0])
        loss_cls_pitch = cls_criterion(pred_pitch, bins_angles[:, 1])        
        loss_cls_roll = cls_criterion(pred_roll, bins_angles[:, 2])
        
        # get the regression values for the output of the model
        reg_yaw = get_reg_form_cls(pred_yaw, prop_func)
        reg_pitch = get_reg_form_cls(pred_pitch, prop_func)
        reg_roll = get_reg_form_cls(pred_roll, prop_func)
        
        # computer regression losses 
        loss_reg_yaw = reg_criterion(reg_yaw, angles[:, 0])
        loss_reg_pitch = reg_criterion(reg_pitch, angles[:, 1])        
        loss_reg_roll = reg_criterion(reg_roll, angles[:, 2])
        
        # total losses 
        loss_yaw = loss_cls_yaw + alpha * loss_reg_yaw
        loss_pitch = loss_cls_pitch + alpha * loss_reg_pitch        
        loss_roll = loss_cls_roll + alpha * loss_reg_roll
        
        loss = loss_yaw + loss_pitch + loss_roll
        
        # compute total loss
        total_loss += loss.cpu().detach().numpy()
        total_loss_perAng[0] += loss_yaw.cpu().detach().numpy() * batch_size
        total_loss_perAng[1] += loss_pitch.cpu().detach().detach().numpy() * batch_size
        total_loss_perAng[2] += loss_roll.cpu().detach().numpy() * batch_size
        
        # compute Accuracy
        total_acc_perAng[0] += torch.sum(torch.abs(reg_yaw - angles[:, 0])).cpu().detach().numpy()
        total_acc_perAng[1] += torch.sum(torch.abs(reg_pitch - angles[:, 1])).cpu().detach().numpy()
        total_acc_perAng[2] += torch.sum(torch.abs(reg_roll - angles[:, 2])).cpu().detach().numpy()
        
        # print results and save to logger
        if i % disp_every == 0:
            print(f'Processing in Itr[{i}/{len(val_loader)}], \
            Losses: Yaw {loss_yaw:.2f}, Pitch {loss_pitch:.2f}, Roll {loss_roll:.2f}')
            
            
    # finish epoch
    total_acc_perAng = [x/dataset_size for x in total_acc_perAng]
    total_loss_perAng = [x/dataset_size for x in total_loss_perAng]
    total_loss /= dataset_size
    print(f'\nEpoch[{epoch_num}] Total Loss: {total_loss}\n\
    Losses Per Angle: Yaw {total_loss_perAng[0]:.2f}, Pitch {total_loss_perAng[1]:.2f}, Roll {total_loss_perAng[2]:.2f}\n\
    Acc Per Angle: Yaw {total_acc_perAng[0]:.2f}, Pitch {total_acc_perAng[1]:.2f}, Roll {total_acc_perAng[2]:.2f}')

    # add to logger 
    logger.add_scalar('Total_Val_Losses/total_loss', total_loss, global_step=epoch_num)
    # losses per angle
    logger.add_scalar('Total_Val_Losses/Yaw', total_loss_perAng[0],global_step=epoch_num)
    logger.add_scalar('Total_Val_Losses/Pitch', total_loss_perAng[1],global_step=epoch_num)        
    logger.add_scalar('Total_Val_Losses/Roll', total_loss_perAng[2],global_step=epoch_num)
    #acc per angle
    logger.add_scalar('Total_Val_Acc/Yaw', total_acc_perAng[0],global_step=epoch_num)
    logger.add_scalar('Total_Val_Acc/Pitch', total_acc_perAng[1],global_step=epoch_num)
    logger.add_scalar('Total_Val_Acc/Roll', total_acc_perAng[2],global_step=epoch_num)
    
    return total_loss, sum(total_acc_perAng)/3


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--use_gpu', dest='use_gpu', help='use gpu during training',
            default=True, type=bool)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=25, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='dataset/300W_LP/', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='dataset/300W_LP/filename_list', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=1.0, type=float)
    parser.add_argument('--checkpoint', dest='checkpoint', help='Path of model checkpoint.',
          default='', type=str)

    args = parser.parse_args()
    return args

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    args = parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size

    # create checkpoints file
    checkpoints_path = 'output/checkpoints'
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # create logger file
    logger = Logger('log_dir')

    # ResNet50 structure
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if args.checkpoint == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')) # load ImageNet weights
    else:
        saved_state_dict = torch.load(args.checkpoint)
        model.load_state_dict(saved_state_dict)

    print('Loading data.') 
    '''
    Data preperation
    '''
    data_dir = args.data_dir
    filename_list_train = args.filename_list + '_train.txt'
    filename_list_val = args.filename_list + '_val.txt'

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # dataset transformer
    transformations = {}
    transformations['train'] = transforms.Compose([transforms.Resize(240),
                                        transforms.RandomCrop(224), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transformations['val'] = transforms.Compose([transforms.Resize(240),
                                        transforms.CenterCrop(224), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # create training dataloader
    pose_train = Pose_300W_LP(data_dir, filename_list_train, transform=transformations['train'], augment=True)
    train_loader = DataLoader(pose_train, batch_size=batch_size, shuffle=True)

    # create val dataloader
    pose_val = Pose_300W_LP(data_dir, filename_list_val, transform=transformations['val'], augment=False)
    val_loader = DataLoader(pose_val, batch_size=batch_size, shuffle=False)

    # transfer model the required device 
    model.to(device)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).to(device)
    # loss for softmax prediction
    cls_criterion = nn.CrossEntropyLoss().to(device)
    #loss of Regression fine tuning
    reg_criterion = nn.MSELoss().to(device)

    # optimizer, set three diffrent lr through the network
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                {'params': get_non_ignored_params(model), 'lr': args.lr},
                                {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                lr = args.lr)

    print('Ready to train network.')

    for epoch in range(num_epochs):

        # train for one epoch
        train_epoch(model, softmax, optimizer, train_loader, logger, epoch, 
                    cls_criterion, reg_criterion, 10)
        # save last checkpoint
        checkpoint = {'optim_dic':optimizer.state_dict(),
                    'state_dic':model.state_dict(),
                    'epoch':epoch
                    }
        torch.save(checkpoint, f'{checkpoints_file}/headpose_last.pth')
        
        # evaluate this epoch
        total_loss, total_acc = evaluate(model, softmax, val_loader, logger, epoch, 
                cls_criterion, reg_criterion, 10)
        # save best checkpoint
        if total_loss < best_loss:
            # save best loss checkpoint
            checkpoint_val = {'optim_dic':optimizer.state_dict(),
                        'state_dic':model.state_dict(),
                        'epoch':epoch,
                        'val_loss':total_loss,
                        'score': total_acc
                        }
            best_loss = total_loss
            torch.save(checkpoint_val, f'{checkpoints_file}/headpose_best.pth')
            print('found better loss')

        if total_acc > best_acc:
            # save best score checkpoint
            checkpoint_val_acc = {'optim_dic':optimizer.state_dict(),
                        'state_dic':model.state_dict(),
                        'epoch':epoch,
                        'val_loss':total_loss,
                        'score': total_acc
                        }
            best_acc = total_acc
            torch.save(checkpoint_val_acc, f'{checkpoints_file}/headpose_best_acc.pth')
            print('found better acc')
