import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import time
import statistics

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--use_gpu', help='Use GPU or not',
            default=True, type=bool)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../checkpoints_loc/headpose_best.pth', type=str)
    parser.add_argument('--video', dest='video', help='Path of video to be predicted, if not provided it will use computer camera.',
          default='', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'

    print(f'cuda status is {torch.cuda.is_available()}')

    batch_size = 1
    snapshot_path = args.snapshot

    tick = time.time()
    # ResNet50 structure
    model = hopenet.Hopenet_Loc(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict['state_dic'])

    tock = time.time()
    #print(f'model loading time is {tock-tick}')

    print ('Loading data.')

    transformations = transforms.Compose([transforms.Resize(300), 
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.to(device)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    # veadio reater form the camera
    use_live = 0 if 
    video = cv2.VideoCapture(0)

    time_frame = []

    while True:

        tick = time.time()
        # Grab a single frame of video
        ret, frame = video.read()
        labels = []

        tock = time.time()
        #print(f'face detection inferance time is {tock-tick}')

        tick = time.time()

        '''
        start the detection for the frame
        '''


        # head pose estimation

        # Transform
        img = Image.fromarray(frame)
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = img.to(device)

        yaw, pitch, roll, x_min, y_min, x_max, y_max = model(img)

        # transform dims 
        x_min *= (frame.shape[1]/ img.shape[3])
        y_min *= (frame.shape[0]/ img.shape[2])
        x_max *= (frame.shape[1]/ img.shape[3])
        y_max *= (frame.shape[0]/ img.shape[2])
        #print(x_min, y_min, x_max, y_max)

        #print(img.shape)

        # limit the values
        x_min = int(max(x_min, 0)); y_min = int(max(y_min, 0))
        x_max = int(min(frame.shape[1], x_max)); y_max = int(min(frame.shape[0], y_max))

        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,0,0),2)

        yaw_predicted = F.softmax(yaw, dim = 1)
        pitch_predicted = F.softmax(pitch, dim = 1)
        roll_predicted = F.softmax(roll, dim = 1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        # draw the head pose estimation axes
        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = (y_min - y_max)/2)
        

        tock = time.time()
        #print(f'inferance time for one frame is {tock-tick}')
        #time_frame.append(tock-tick)
        # output the prediction
        cv2.imshow('Emotion Detector',frame)
        # check quit conditions 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    #print(time_frame)
    mean_time = statistics.mean(time_frame)
    std_time = statistics.stdev(time_frame)
    print(f'interface mean time{mean_time}')
    print(f'interface std time{std_time}')

    video.release()
    cv2.destroyAllWindows()
