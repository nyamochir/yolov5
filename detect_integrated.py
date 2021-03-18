import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np 
import tensorflow as tf
import sys


original_scale = (640, 384) # rescaleing the input images --> 
model_index_file = {
    "0": "single_frame/multiple_frames/ypol23.names",
    "1": "single_frame/multiple_frames/ypol18_nonfood.names"
}
#model21prod = attempt_load(weights1, map_location=device)
#model16prod = attempt_load(weights2, map_location=device)
device = "cuda"
# best weights  --> 
weights = ["weights/products21new/best.pt", "weights/products16new/best.pt"]

def models_load(weights):
    
    """
    load the models into the GPU -->  
    
    """
    
    loaded_models = []
    for weight in weights:
        #device = "cuda"
        cmodel = attempt_load(weight, map_location=device)
        loaded_models.append(cmodel)
        
    return loaded_models



#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def get_tensor(img):
    
    
    """
        image or the single frame of the data data stream or the image stream --> 
        return sth the shape that can be used for GPU Computation.
    
    """
    print("tensor conversion")
    img = cv2.resize(img, (640, 384), interpolation = cv2.INTER_AREA)
    img = img/255.0
    img =torch.from_numpy(img)

    img = torch.reshape(img, [1, img.shape[2], img.shape[0], img.shape[1]])
    img = img.to(device = "cuda", dtype = torch.float)
    return img

def get_prediction(img, current_model):
    """
    gets the model and prediction frame
    
    """
    print("prediction generation ...")
    #print(current_model.names) # names of the current model ---> 
    current_pred = current_model(img, augment = False)
    return current_pred

def apply_nms(current_pred, nms = 0.2, iou = 0.1):
    """
    need to transfer entire prediction --> 
    
    then apply nms with iou and nms thresholds --> to limit the boundaries --> 
    
    default 
    nms = 0.2
    iou = 0.1 --> change it --> 
    
    """
    print("nms applying ...")
    pred = non_max_suppression(current_pred[0], nms, iou, None, False) 

    return pred # nms applied prediction results 


def get_labels(pred_nms_applied, target_label_path):
    
    
    """
    
    pred_nms_applied --after the stage of the applying the nms -->  current prediction will 
    
    be used for 
    
    
    """
    print("label parsing ...")
    #GPU assigned --> elements --> 
    boundaries = pred_nms_applied[0][:, :4] 
    classes = pred_nms_applied[0][:, 5]# nms applied --> 
    #classes
    
    # CPU assignment -->  since some applications will not able to integrated --> 
    cpu_boundaries= boundaries.to("cpu") # 
    
    cpu_classes = classes.to("cpu") # let the boundaries and classes stay as GPU garbage
    
    
    index = 0
    #cpu_class =classes.to("cpu") 
    w = original_scale[0]
    h = original_scale[1]
    limit = 0
    labels = ""
    for boundary in cpu_boundaries:
        x1 =  int(boundary[0])
        y1=   int(boundary[1])
        x2 =  int(boundary[2])
        y2 =  int(boundary[3])

        centerx = (x1+x2)/2/w
        centery = (y1+y2)/2/h
        width = (x1 - centerx)/w
        height = (y1 - centery)/h  

        #centerx
        
        # absolute value generation almost not need to be used






        #
        #x1 = str(/w)
        #y1 =  str(abs(int(boundary[1]))*h)
        #x2 = str(abs(int(boundary[2]))/w)
        #y2 =  str(abs(int(boundary[3]))*h)
        #class = classes[index].to("cpu")
        class_name = str(int(cpu_classes[index]))
        label = class_name+" "+ str(centerx)+" "+ str(centery)+" "+ str(width)+" "+str(height)+"\n"
        labels+=label
        #if limit == 10: # there can be many predictions --> 

        #   break
        #limit+=1
    with open(target_label_path, "w") as file:
        file.write(labels)
        
        
    return labels # deconstruct on the real time video


    #print(class_name, x1, y1, x2, y2)
    
    
def detect_in_images(loaded_models):
    
    image_path = "single_frame/multiple_frames/images/"
    label_path = "single_frame/multiple_frames/labels"
    
    for image_name in os.listdir(image_path):
        model_index = 0
        label_name = image_name.split(".")[0] + ".txt"
        
        target_label_path = label_path + str(model_index)+"/"+label_name
        target_image_path = image_path + image_name
        
        
        
        for current_model in loaded_models:
            try:
                os.mkdir(label_path + str(model_index))
            except:
                pass
            print("Current running model:", model_index, "\n# classes:", len(current_model.names), "\nClass Names : ", current_model.names)
            start =time.time()
            img = cv2.imread(target_image_path)
            img = get_tensor(img)
            current_pred = get_prediction(img, current_model)

            #
            pred_nms_applied = apply_nms(current_pred, 0, .1) # for sake of the faster implementation-->

            get_labels(pred_nms_applied, target_label_path)


            #print("Current tensor shape:",  img.shape)
            print("single frame inference time ", time.time() - start)
            model_index+=1
            
            
    
            
    
def detect_in_videos():
    
    
    pass
    
    



if __name__ == "__main__":


    # loading the avialable models -->
    # first stage of the model --> 
    
    # models --> loaded --> 
    loaded_models = models_load(weights)
    
    #model_index = 0 
    
    
    if sys.argv[-1] == "image":
        detect_in_images(loaded_models)
    if sys.argv[-1] == "video":
        detect_in_images(loaded_models)
    
    
    
    """
    
    for current_model in loaded_models:
        print("Current running model:", model_index, "\n# classes:", len(current_model.names), "\nClass Names : ", current_model.names)
        start =time.time()
        img = cv2.imread("single_frame/cam_1_965.jpg")
        img = get_tensor(img)
        current_pred = get_prediction(img, current_model)

        #
        pred_nms_applied = apply_nms(current_pred, 0, .1) # for sake of the faster implementation-->

        get_labels(pred_nms_applied)
        
        
        print("Current tensor shape:",  img.shape)
        print("single frame inference time ", time.time() - start)
        model_index+=1
    """