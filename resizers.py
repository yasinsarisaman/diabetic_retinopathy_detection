# -- coding: utf-8 --
"""
Created on Sun May 15 20:44:36 2022

@author: Yasin
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

datasetDir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST"

imgSize = 150

classes = ["NO_DR" , "MILD_DR" , "MODERATE_DR" , "SEVERE_DR" , "PROLIFERATIVE_DR"]

no_dr_dir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST/TEST_RESIZED/NO_RESIZED/"
mild_DR_dir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST/TEST_RESIZED/MILD_RESIZED/"
moderate_DR_dir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST/TEST_RESIZED/MODERATE_RESIZED/"
severe_DR_dir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST/TEST_RESIZED/SEVERE_RESIZED/"
proliferative_DR_dir = "C:/Users/yasin/Desktop/BitirmeProjesi/DATASET/DR_DATA/TEST/TEST_RESIZED/PROLIFERATIVE_RESIZED/"

resizedTrainData = []
resizedTestData = []


def createTrainData():
    for category in classes:
        path = os.path.join(datasetDir, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                dataArrayTrain = cv2.imread(os.path.join(path , img), cv2.IMREAD_ANYCOLOR)
                resizedTrainData.append(cv2.resize(dataArrayTrain, (imgSize, imgSize)))
                imgNew = cv2.resize(dataArrayTrain, (imgSize, imgSize))
                if class_num==0:
                    cv2.imwrite(no_dr_dir + img, imgNew)
                    print ("0")
                elif class_num==1:
                    cv2.imwrite(mild_DR_dir+ img, imgNew)
                    print ("01")
                elif class_num==2:
                    cv2.imwrite(moderate_DR_dir+ img, imgNew) 
                    print ("02")
                elif class_num==3:
                    cv2.imwrite(severe_DR_dir+ img, imgNew)
                    print ("03")
                elif class_num==4:
                    cv2.imwrite(proliferative_DR_dir+ img, imgNew) 
                    print ("04")
                    
            except Exception as e:
                pass
            
            
createTrainData()