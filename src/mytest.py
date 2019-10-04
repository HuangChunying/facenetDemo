# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:26:58 2019

@author: train
"""

import argparse
import os
import file
import loadImage
import tensorflow as tf
import numpy as np
import time

import facenet

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            first = True  
            print ("%s: %s" % ("loading:", time.ctime(time.time())))    
            
            
            dict_0=InputPara(parse_arguments(" "),first)
                    
            loadImage.loadImage(dict_0,first)
            first=False
            facenet.load_model("../../model/20180402-114759/20180402-114759.pb")
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")            
            
            print ("%s: %s" % ("start:", time.ctime(time.time()))) 
            while True:                
                temppathDir =  os.listdir("../tempImage")
                if(len(temppathDir)>0):            
                    child = os.path.join('%s%s%s' % ("../tempImage","/", temppathDir[0]))
                    mainImage=[child]
                    
                    dict_1=InputPara(parse_arguments(mainImage),first)                  
                    dict_loadImage=loadImage.loadImage(dict_1,first)

                    
                    feed_dict = { images_placeholder: dict_loadImage["images"], phase_train_placeholder:False }
                    dict_loadImage["embeddings"]=embeddings
                    dict_loadImage["feed_dict"]=feed_dict
                    
                    emb = sess.run(dict_loadImage["embeddings"], dict_loadImage["feed_dict"])          
                    nrof_images = len(dict_loadImage["image_files"])
                    Alldist=[]
                    for j in range(nrof_images):
                        dist = np.sqrt(np.sum(np.square(np.subtract(emb[-1,:], emb[j,:]))))
                        Alldist.append(dist);
                    tempmin=min(Alldist)
                    Alldist.remove(tempmin)
                        #print(Alldist)
                    tempmin2=min(Alldist)
                    resindex=Alldist.index(tempmin2)
                        #print(resindex+1)
                    print(dict_loadImage["image_files"][resindex])
                    if(not first):
                        loadImage.image_files_list.pop()
                        loadImage.img_list.pop()
                    
                    os.remove(child)
                    print ("%s: %s" % ("end", time.ctime(time.time())))
                else:
                    time.sleep(5)
                    print ("%s: %s" % ("alive--", time.ctime(time.time())))
            


def InputPara(args,first): 
    if (first):     
        filepath='../emp'
        allFileName = file.eachFile(filepath)
        args.image_files=[];
        for x in allFileName: 
             args.image_files.append(x)
        
    args.model="../../model/20180402-114759/20180402-114759.pb"
        
    dict_1={"image_files":args.image_files,"image_size":args.image_size,"margin":args.margin,"gpu_memory_fraction":args.gpu_memory_fraction,"model":args.model}
    return dict_1
    
          
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)




if __name__ == '__main__':
    main()
 

    
