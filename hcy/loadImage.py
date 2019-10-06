# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:26:58 2019

@author: train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import numpy as np
import os
import copy
import facenet
import align.detect_face




 # 检测图片中的人脸　返回人脸所在矩形区域       
def loadImage_detectFace(dictInfo,pnet, rnet, onet): 
        

    img_list = load_and_align_data(dictInfo["image_files"], dictInfo["image_size"], dictInfo["margin"], dictInfo["gpu_memory_fraction"],pnet, rnet, onet)        
    images = np.stack(img_list)
    dictInfo["images"]=images    
    return dictInfo
            # Run forward pass to calculate embeddings
      
            
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction,pnet, rnet, onet):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
#    print('Creating networks and loading parameters')
#    with tf.Graph().as_default():
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#        with sess.as_default():
#            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    number_img = len(tmp_image_paths)
    curNum = 0
    for image in tmp_image_paths:
        curNum +=1
        
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
        if(curNum % 50 == 0):        
            print("current loading",curNum,"of",number_img)
    #images = np.stack(img_list)
    #return images
    return img_list
           
            

