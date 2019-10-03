# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:26:58 2019

@author: train
"""

import compare
import os
import file
import time
import loadImage
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face




def main(args,first):
    
    print(args)
   
    if (first):
       
        filepath='../emp'
        allFileName = file.eachFile(filepath)
        
        for x in allFileName: 
             args.image_files.append(x)
        
        args.model="../../model/20180402-114759/20180402-114759.pb"
        print(args)
    image_files=args.image_files
    image_size=args.image_size
    margin=args.margin
    gpu_memory_fraction=args.gpu_memory_fraction
    model=args.model

    images,image_files_list, image_size, margin, gpu_memory_fraction,model=loadImage.loadImage(image_files, image_size, margin, gpu_memory_fraction,model,first)

    with tf.Graph().as_default():
        image_files,embeddings,feed_dict=loadImage.main(images,image_files_list, image_size, margin, gpu_memory_fraction,model)
        with tf.Session() as sess:
            emb = sess.run(embeddings, feed_dict)          
            nrof_images = len(image_files)
            Alldist=[]
            for j in range(nrof_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[j,:]))))
                Alldist.append(dist);
            tempmin=min(Alldist)
            Alldist.remove(tempmin)
                #print(Alldist)
            tempmin2=min(Alldist)
            resindex=Alldist.index(tempmin2)
                #print(resindex+1)
            print(image_files[resindex+1])



if __name__ == '__main__':
 
    temppathDir =  os.listdir("../tempImage")
    if(len(temppathDir)>0):            
        child = os.path.join('%s%s%s' % ("../tempImage","/", temppathDir[0]))
        mainImage=[child]
        first = True
        main(compare.parse_arguments(mainImage),first)
#        first = False
#        main(compare.parse_arguments(mainImage),first)
            
#    while True:    
#        temppathDir =  os.listdir("../tempImage")
#        if(len(temppathDir)>0):            
#            child = os.path.join('%s%s%s' % ("../tempImage","/", temppathDir[0]))
#            mainImage=[child]
#            main(compare.parse_arguments(mainImage))
#            os.remove(child)
#        time.sleep(5)
#        print ("%s: %s" % ("nu;;", time.ctime(time.time())))
    
