# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:42:08 2019

@author: DG
"""
#import Libraries
import cv2
import time
import numpy as np

#Path for Label list and read it in labels
labels_path = 'coco.names'
labels = open(labels_path).read().split("\n")

#define confidence level and threshold
pre_conf = 0.5
threshold = 0.3

#Define different random colors for boxes
np.random.seed(0)
colors = np.random.randint(0,255,size=(len(labels),3),dtype='uint8')

#path for YOLO_wightss and config file
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'

#Define neural net
net = cv2.dnn.readNetFromDarknet(config_path,weights_path)

#read image for detect objects and find height,width
img = cv2.imread("2.jpg")
(H,W) = img.shape[:2]

#Find YOLO Output layers 
ln = net.getLayerNames()

ln =[ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#give image as input to neural network
blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),swapRB=True,crop=False)
net.setInput(blob)
start = time.time()
out_lay = net.forward(ln)
end = time.time()

print("Yolo_Time_length:{:.6f}".format(end-start))

#List of boxes,classIds,confidences
boxes = []
classIds =[]
confidences = []

#Identify best prediction and draw bounding boxes
for output in out_lay:
    for detection in output:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence>pre_conf:
            box = detection[0:4]*np.array([W,H,W,H])
            (cenX,cenY,width,height) = box.astype('int32')
            
            x = int(cenX-(width/2))
            y = int(cenY-(height/2))
            
            boxes.append([x,y,int(width),int(height)])
            confidences.append(float(confidence))
            classIds.append(classId)

#Used to suppress meaning remove weak boxes
index = cv2.dnn.NMSBoxes(boxes,confidences,pre_conf,threshold)

if len(index)>0:
    for i in index.flatten():
        (x,y) = (boxes[i][0],boxes[i][1])
        (w,h) = (boxes[i][2],boxes[i][3])
        
        color = [int(c) for c in colors[classIds[i]]]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        text = print("{}:{:.4f}".format(labels[classIds[i]],confidences[i]))
        cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_DUPLEX,0.5,color,2)


cv2.imshow("image",cv2.resize(img,(500,500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

