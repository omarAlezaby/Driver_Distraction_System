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
    parser.add_argument('--use_gpu', dest='use_gpu', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../checkpoints/headpose_last.pth', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='haarcascade_frontalface_default.xml', type=str)
    parser.add_argument('--video', dest='video', help='Path of video to be predicted, if not provided it will use computer camera.',
          default='', type=str)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    print(f'cuda status is {torch.cuda.is_available()}')

    batch_size = 1
    use_gpu = args.use_gpu
    snapshot_path = args.snapshot
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    tick = time.time()
    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict['state_dic'])

    tock = time.time()
    #print(f'model loading time is {tock-}')
    
    tick = time.time()
    # opencv face detection model
    cnn_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    tock = time.time()
    #print(f'face detection model loading time is {tock-tick}')

    print ('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.to(device)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)

    # veadio reater form the camera
    use_camera = 0 if args.video == '' else args.video
    video = cv2.VideoCapture(use_camera)

    time_frame = []

    while True:

        tick = time.time()
        # Grab a single frame of video
        ret, frame = video.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cnn_face_detector.detectMultiScale(gray,1.3,5)

        tock = time.time()
        #print(f'face detection inferance time is {tock-tick}')

        tick = time.time()
        for idx, (x,y,w,h) in enumerate(faces):

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # Get x_min, y_min, x_max, y_max, conf
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            # head pose estimation
            tick_inner = time.time()
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = int(max(x_min, 0)); y_min = int(max(y_min, 0))
            x_max = int(min(frame.shape[1], x_max)); y_max = int(min(frame.shape[0], y_max))
            # Crop image
            img = frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).to(device)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

            # draw the head pose estimation axes
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            
            tock_inner = time.time()
            print(f'inference time for one face detection is {tock_inner-tick_inner}')

        tock = time.time()
        print(f'inferance time for one frame is {tock-tick}')
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
