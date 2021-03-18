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

nms = float(sys.argv[-3])
iou = float(sys.argv[-2])
print(nms, iou)
original_scale = (640, 384) # rescaleing the input images --> 

#model21prod = attempt_load(weights1, map_location=device)
#model16prod = attempt_load(weights2, map_location=device)
device = "cuda"
# best weights  --> 
# curretly 3
weights = ["weights/latest/product21/best.pt", "weights/latest/product16/best.pt", "weights/latest/productwework/best.pt"]
model_names = ["YP_Food21", "YP_Nonfood16", "YP_WeWork"]

def get_dict(model_num):
    model_index = str(model_num)
    
    model_index_file = {
    "0": "index_files/ypol_food21.names",
    "1": "index_files/ypol_nonfood.names",
    "2":"index_files/ypol_wework.names"
    }
    with open(model_index_file[model_index]) as file:
        key_dict = {}
        index = 0
        for line in file.readlines():
            if "\n" in line:
                line = line[:-1]
            key_dict[str(index)]= line
            index+=1
        #print(key_dict)
    return key_dict

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
    #print("prediction generation ...")
    #print(current_model.names) # names of the current model ---> 
    current_pred = current_model(img, augment = False)
    return current_pred

def apply_nms(current_pred):
    """
    need to transfer entire prediction --> 
    
    then apply nms with iou and nms thresholds --> to limit the boundaries --> 
    
    default 
    nms = 0.2
    iou = 0.1 --> change it --> 
    
    """
    #print("nms applying ...")
    pred = non_max_suppression(current_pred[0], nms, iou, None, False) 

    return pred # nms applied prediction results 


def get_labels(pred_nms_applied, target_label_path, image):
    
    
    """
    
    pred_nms_applied --after the stage of the applying the nms -->  current prediction will 
    
    be used for 
    
    
    """
    #print("label parsing ...")
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
    if image:
        with open(target_label_path, "w") as file:
            file.write(labels)
        

        
    return labels # deconstruct on the real time video


    #print(class_name, x1, y1, x2, y2)
    
def detect_in_frame(loaded_models, img, image = False): # default image as false 
    """
    img  -- frame
    
    """
    
    #image_path = "single_frame/multiple_frames/videos/"
    #label_path = "single_frame/multiple_frames/labels"
    
    #for image_name in os.listdir(image_path):
    #    model_index = 0
    #    label_name = image_name.split(".")[0] + ".txt"
        
    #    target_label_path = label_path + str(model_index)+"/"+label_name
    #    target_image_path = image_path + image_name
    target_label_path = ""
    if not image:
            target_label_path = ""
        
    model_index = 0
    h, w, d = img.shape # frame shape will be used --> 
    
    #frame = img.copy()
    frame_versions = []
    for i in range(len(loaded_models)):
        frame_versions.append(img.copy())
    for current_model in loaded_models:
            #try:
            #    os.mkdir(label_path + str(model_index))
            #except:
            #    pass
            frame = frame_versions[model_index]
            #print("Current running model:", model_index, "\n# classes:", len(current_model.names), "\nClass Names : ", current_model.names)
            start =time.time()
            
            #img = cv2.imread(target_image_path)
            frame = get_tensor(frame)
            current_pred = get_prediction(frame, current_model)

            
            pred_nms_applied = apply_nms(current_pred) # for sake of the faster implementation-->
            #print(pred_nms_applied.shape)
            
            labels = get_labels(pred_nms_applied, target_label_path, image)
            #print(len(labels.split("\n")))
            lineType               = 2
            fontScale              = 1
            fontColor              = (255,255,255)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            
            
            key_dict = get_dict(model_index)
            #print(key_dict)
            each_rows =  labels.split("\n")
            print("Predicted labels: ", len(each_rows))
            for line in each_rows:
                            if "\n" in line:
                                line = line[:-1]
                            #print(line)
                            
                            
                            line = line.split(" ")
                            #line =  line.split(" ")
                            try:
                                index  = line[0]
                                #print(label_path+current_label_name, index, len(line))
                                x = (float(line[1]), float(line[2]))# tuples 
                                y = (float(line[3]), float(line[4]))
                                #print(x, y)
                                #print(key_dict[str(index)], x, y)#

                                x1 = int((x[0] - y[0])*w)
                                y1 = int((x[1] - y[1])*h)
                                x2 = int((x[0] + y[0])*w)
                                y2 = int((x[1] + y[1])*h)
                                #print(x1, y1, x2, y2)
                                img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0) ,2)
                                img = cv2.putText(img, key_dict[index]+"**model:"+model_names[model_index]+"***", 
                                (x1, y1),# inside the box there will the text --> 
                                font, 
                                fontScale,
                                fontColor,
                                lineType)
                            except:
                                pass
             

            #print("Current tensor shape:",  img.shape)
            print("Single frame inference time ", time.time() - start)
            model_index+=1
    return img
    
    
def detect_in_images(loaded_models, image = True):
    
    """
    default value of the image -- > 
    
    --------
    
    
    
    """
    
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
            #print("Current running model:", model_index, "\n# classes:", len(current_model.names), "\nClass Names : ", current_model.names)
            start =time.time()
            img = cv2.imread(target_image_path)
            img = get_tensor(img)
            current_pred = get_prediction(img, current_model)

            #
            pred_nms_applied = apply_nms(current_pred) # for sake of the faster implementation-->

            labels = get_labels(pred_nms_applied, target_label_path, image)


            #print("Current tensor shape:",  img.shape)
            print("single frame inference time ", time.time() - start)
            model_index+=1
            
    
def detect_in_videos(loaded_models, target_path, video_name):
    
    
    
    cap = cv2.VideoCapture(target_path+video_name)
    # extract key information related to the  keys --> 
    cap1  = cv2.VideoCapture(target_path+video_name)
    status, init_frame = cap1.read() # just read first frame
    h, w, d = 0, 0, 0
    if status:
        h, w, d = init_frame.shape
    
    #print(h, w, d)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    try:
        os.mkdir(target_path+"annotated_videos")
    except:
        pass
    out = cv2.VideoWriter(target_path+'/annotated_videos/'+video_name, fourcc, 20.0, (w, h))
    index_number = 0
    while True:
        
        status, frame = cap.read()
        
        
        if status: # successful load for the video --> 
            
            #h, w, d = frame.shape
            
            frame = detect_in_frame(loaded_models, frame)
            out.write(frame)
            
            
        else:
            break
        
            
            
    
        #try:
                
                #cv2.imshow(video_name.split(".")[0], frame)
                
        #except:
        #        print(target_path+video_name, " printing Encountered error at Frame Number ",  index_number)
        #        break        
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # testing limiter
            #if index_number == 10:
            #    break
        
        index_number+=1
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        
    
    




if __name__ == "__main__":


    # loading the avialable models -->
    # first stage of the model --> 
    
    # models --> loaded --> 
    loaded_models = models_load(weights)
    
    #model_index = 0 
    
    
    if sys.argv[-1] == "image":
        detect_in_images(loaded_models)
    if sys.argv[-1] == "video":
        
        videos_path = "video_test/"
        for video_name in os.listdir(videos_path):
            if ".mp4" in video_name:
                detect_in_videos(loaded_models, videos_path, video_name)
            
    
        
    
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