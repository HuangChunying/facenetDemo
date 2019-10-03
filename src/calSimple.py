from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import loadImage
import mytest

def calculuate():
    image_files, image_size, margin, gpu_memory_fraction,model,first=mytest.main()
    images,image_files_list, image_size, margin, gpu_memory_fraction,model=loadImage.loadImage(image_files, image_size, margin, gpu_memory_fraction,model,first)
    image_files,embeddings,feed_dict=loadImage.mian( images,image_files_list, image_size, margin, gpu_memory_fraction,model)
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
    calculuate()
            
            

