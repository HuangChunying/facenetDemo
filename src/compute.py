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

def calculuate(image_files,embeddings,feed_dict):
        with tf.Session() as sess:
      
            # Load the model
#            facenet.load_model(args.model)
#            
#            # Get input and output tensors
#            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#            
#            images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
#            # Run forward pass to calculate embeddings
#            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict)
            
            nrof_images = len(image_files)

#            print('Images:')
#            for i in range(nrof_images):
#                print('%1d: %s' % (i, image_files[i]))
#            print('')

            # Print distance matrix
#            print('Distance matrix')
#            print('    ', end='')
#            for i in range(nrof_images):
#                print('    %1d     ' % i, end='')
#            print('')
#            for i in range(nrof_images):
#                print('%1d  ' % i, end='')
#                for j in range(nrof_images):
#                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#                    print('  %1.4f  ' % dist, end='')
#                print('')
                
                
#            print('================<><><>')
#            print('    ', end='')
#            for i in range(nrof_images):
#                print('    %1d     ' % i, end='')
#            print('')
#            print('main', end='')
            Alldist=[]
            for j in range(nrof_images):
                dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[j,:]))))
                Alldist.append(dist);
                #print('  %1.4f  ' % dist, end='')
#            print('')
#            print('-----------------end---')
            #print(Alldist)
            tempmin=min(Alldist)
            Alldist.remove(tempmin)
            #print(Alldist)
            tempmin2=min(Alldist)
            resindex=Alldist.index(tempmin2)
            #print(resindex+1)
            print(image_files[resindex+1])
            
            
#def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
#
#    minsize = 20 # minimum size of face
#    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
#    factor = 0.709 # scale factor
#    
#    print('Creating networks and loading parameters')
#    with tf.Graph().as_default():
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#        with sess.as_default():
#            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
#  
#    tmp_image_paths=copy.copy(image_paths)
#    img_list = []
#    for image in tmp_image_paths:
#        img = misc.imread(os.path.expanduser(image), mode='RGB')
#        img_size = np.asarray(img.shape)[0:2]
#        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
#        if len(bounding_boxes) < 1:
#          image_paths.remove(image)
#          print("can't detect face, remove ", image)
#          continue
#        det = np.squeeze(bounding_boxes[0,0:4])
#        bb = np.zeros(4, dtype=np.int32)
#        bb[0] = np.maximum(det[0]-margin/2, 0)
#        bb[1] = np.maximum(det[1]-margin/2, 0)
#        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
#        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
#        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
#        prewhitened = facenet.prewhiten(aligned)
#        img_list.append(prewhitened)
#    images = np.stack(img_list)
#    return images

