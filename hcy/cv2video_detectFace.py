# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:48:18 2019

@author: train
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import argparse
import tensorflow as tf
import numpy as np
import align.detect_face
import cv2





# 从摄像头读取视频流　并检测人脸(可设置单个脸检测)
def main(args):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    cap = cv2.VideoCapture(0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
    print('size:'+repr(size)) 
    
    while True:
        ret,frame = cap.read()
        img = frame
        img = img[:,:,0:3]
        
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if args.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            
            bb = np.zeros((nrof_faces,4),dtype=np.int32)
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)               
                bb[i][0] = np.maximum(det[0]-args.margin/2, 0)
                bb[i][1] = np.maximum(det[1]-args.margin/2, 0)
                bb[i][2] = np.minimum(det[2]+args.margin/2, img_size[1])
                bb[i][3] = np.minimum(det[3]+args.margin/2, img_size[0])
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2],bb[i][3]), (0, 255, 0), 2)
                cv2.putText(frame, "No Body", (bb[i][0],bb[i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)                
#           
        cv2.imshow("Video",frame)
        c = cv2.waitKey(1)

        if c==27:
            break
    cap.release()
    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
