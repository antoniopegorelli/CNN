# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 14:57:35 2019

@author: anton
"""

import numpy as np

from os import listdir
from os.path import isfile, join

import xlwt

from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3, NASNetLarge, VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions

import matplotlib.pyplot as img_plt


# Creating excel file to store results
wBook = xlwt.Workbook()
sheet1 = wBook.add_sheet("ImageNet")

# Generation excel file header
row = 0
rows = sheet1.row(row)
# header = ["Image","MobileNetV2 Label","MobileNetV2 Percentage","","InceptionV3 Label","InceptionV3 Percentage","","VGG16 Label","VGG16 Percentage"]
header = ["Image","ResNet50 Label","ResNet50 Percentage","","NASNetLarge Label","NASNetLarge Percentage","","VGG16 Label","VGG16 Percentage"]

for index, value in enumerate(header):
    rows.write(index, value)
    

# Loading application models
#mobileNet_model = MobileNetV2(weights='imagenet')
ResNet50_model = ResNet50(weights='imagenet')
#inception_model = InceptionV3(weights='imagenet')
NASnet_model = NASNetLarge(weights='imagenet')
vgg16_model = VGG16(weights='imagenet')

# Listing files from folder
test_images = [f for f in listdir('Imagens/') if isfile(join('Imagens/', f))]

# Loop to test images
for filename in test_images:
    
    print('\n\n\n\n-----------------------' + filename + '--------------------------')
    
    # loading the image
    path = 'Imagens/' + filename
    
    # img_original_inc = load_img(path, target_size=(299, 299))
    img_original_NAS = load_img(path, target_size=(331, 331))
    img_original = load_img(path, target_size=(224, 224))
    img_plt.imshow(img_original)
    img_plt.show()

    print("Analyzing picture...")
    
    # Updating excel file
    row += 1
    rows = sheet1.row(row)
    rows.write(0, filename)
    
    # Converting pixels to numpy array
    # img_array_inc = img_to_array(img_original_inc)
    img_array_NAS = img_to_array(img_original_NAS)
    img_array = img_to_array(img_original)
    img_plt.imshow(np.uint8(img_array))
    img_plt.show()
     
    # Reshaping image for the model
    # img_array_inc = img_array_inc.reshape((1, img_array_inc.shape[0], img_array_inc.shape[1], img_array_inc.shape[2]))
    img_array_NAS = img_array_NAS.reshape((1, img_array_NAS.shape[0], img_array_NAS.shape[1], img_array_NAS.shape[2]))
    img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
    
    # # Predicting image with MobileNetV2
    # print("Predicting with MobileNetV2:")
    # predictions = mobileNet_model.predict(img_array)
    # label = decode_predictions(predictions)
    
    # for i in range(0,5):
    #     label_i = label[0][i]
        
    #     # Console print
    #     print('%s (%.2f%%)' % (label_i[1], label_i[2]*100))
        
    #     # Updating excel file
    #     rows = sheet1.row(row+i)
    #     rows.write(1, label_i[1])
    #     rows.write(2, '%.2f%%' % (label_i[2]*100))
        
    # print('\n')
    
    # Predicting image with ResNet50
    print("Predicting with ResNet50:")
    predictions = ResNet50_model.predict(img_array)
    label = decode_predictions(predictions)
    
    for i in range(0,5):
        label_i = label[0][i]
        
        # Console print
        print('%s (%.2f%%)' % (label_i[1], label_i[2]*100))
        
        # Updating excel file
        rows = sheet1.row(row+i)
        rows.write(1, label_i[1])
        rows.write(2, '%.2f%%' % (label_i[2]*100))
        
    print('\n')
    
    # # Predicting image with InceptionV3
    # print("Predicting with InceptionV3:")
    # predictions = inception_model.predict(img_array_inc)
    # label = decode_predictions(predictions)
    
    # for i in range(0,5):
    #     label_i = label[0][i]
        
    #     # Console print
    #     print('%s (%.2f%%)' % (label_i[1], label_i[2]*100))
        
    #     # Updating excel file
    #     rows = sheet1.row(row+i)
    #     rows.write(4, label_i[1])
    #     rows.write(5, '%.2f%%' % (label_i[2]*100))
        
    # print('\n')
    
    print("Predicting with NASNetLarge:")
    predictions = NASnet_model.predict(img_array_NAS)
    label = decode_predictions(predictions)
    
    for i in range(0,5):
        label_i = label[0][i]
        
        # Console print
        print('%s (%.2f%%)' % (label_i[1], label_i[2]*100))
        
        # Updating excel file
        rows = sheet1.row(row+i)
        rows.write(4, label_i[1])
        rows.write(5, '%.2f%%' % (label_i[2]*100))
        
    print('\n')
    
    # Predicting image with VGG16
    print("Predicting with VGG16:")
    predictions = vgg16_model.predict(img_array)
    label = decode_predictions(predictions) 
    
    for i in range(0,5):
        label_i = label[0][i]
        
        # Console print
        print('%s (%.2f%%)' % (label_i[1], label_i[2]*100))
        
        # Updating excel file
        rows = sheet1.row(row+i)
        rows.write(7, label_i[1])
        rows.write(8, '%.2f%%' % (label_i[2]*100))
        
    print('\n')
    row += 5
    
wBook.save("Results.xls")