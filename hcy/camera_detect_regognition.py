# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 11:10:57 2019

@author: train
"""
import argparse
import os
import file
import loadImage
import tensorflow as tf
import numpy as np
import time
import align.detect_face
import cv2
from scipy import misc
from PIL import Image, ImageDraw, ImageFont

import facenet

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            first = True  
            print ("%s: %s" % ("loading model:", time.ctime(time.time())))    
            
            facenet.load_model("../../model/20180402-114759/20180402-114759.pb")
            
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")       
            
            
            
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            sess_facenet = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess_facenet.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess_facenet, None)
            
            print ("%s: %s" % ("loading image hub:", time.ctime(time.time())))    
            dict_0=InputPara(parse_arguments(" "),first)                  
            dict_loadImage_0=loadImage.loadImage(dict_0,pnet, rnet, onet)
            
            feed_dict_0 = { images_placeholder: dict_loadImage_0["images"], phase_train_placeholder:False }
            dict_loadImage_0["embeddings"]=embeddings
            dict_loadImage_0["feed_dict"]=feed_dict_0   
            print ("%s: %s" % ("compute image emb:", time.ctime(time.time())))    
            emb_0 = sess.run(dict_loadImage_0["embeddings"], dict_loadImage_0["feed_dict"])  
            
            first=False

            # Get input and output tensors

            minsize = 20 # minimum size of face
            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
            factor = 0.709 # scale factor
            print("+++++")
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
                        if dict_0["detect_multiple_faces"]:
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
                        nameOfImg = "No body"
                        img_list=[]
                        dict_loadImage={}
                        det = np.squeeze(det)               
                        bb[i][0] = np.maximum(det[0]-dict_0["margin"]/2, 0)
                        bb[i][1] = np.maximum(det[1]-dict_0["margin"]/2, 0)
                        bb[i][2] = np.minimum(det[2]+dict_0["margin"]/2, img_size[1])
                        bb[i][3] = np.minimum(det[3]+dict_0["margin"]/2, img_size[0])
                        cropped = img[bb[i][1]:bb[i][3],bb[i][0]:bb[i][2],:]
                        aligned = misc.imresize(cropped, (dict_0["image_size"], dict_0["image_size"]), interp='bilinear')
                        prewhitened = facenet.prewhiten(aligned)
                        img_list.append(prewhitened)
                        images = np.stack(img_list) 
                        dict_loadImage["images"]=images
                        
                        feed_dict = { images_placeholder: dict_loadImage["images"], phase_train_placeholder:False }
                        dict_loadImage["embeddings"]=embeddings
                        dict_loadImage["feed_dict"]=feed_dict
                        
                        emb = sess.run(dict_loadImage["embeddings"], dict_loadImage["feed_dict"])  
                        emb = np.append(emb_0,emb,axis=0)
                        
                        nrof_images = len(images)+len(dict_loadImage_0["image_files"])
                        Alldist=[]
                        for j in range(nrof_images):
                            dist = np.sqrt(np.sum(np.square(np.subtract(emb[-1,:], emb[j,:]))))
                            Alldist.append(dist);
                        tempmin=min(Alldist)
                        Alldist.remove(tempmin)
                            #print(Alldist)
                        tempmin2=min(Alldist)
                        if(tempmin2 < 1):
                            resindex=Alldist.index(tempmin2)
                            nameOfImg=dict_loadImage_0["image_files"][resindex].split("/")[-1].split(".")[0]
                            print(nameOfImg)
                        else:
                            nameOfImg="sos"
                            print("NO RESULT !!!")
                                            
                        np.delete(emb,-1,axis=0)
                        print ("%s: %s" % ("end", time.ctime(time.time())))                       
                        
                        
                        
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2],bb[i][3]), (0, 255, 0), 2)
                        #cv2.putText(frame, nameOfImg, (bb[i][0],bb[i][1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)                
                        frame=cv2ImgAddText(frame, nameOfImg, bb[i][0], bb[i][1], (0,255,0), 20)
                cv2.imshow("Video",frame)
                c = cv2.waitKey(1)
        
                if c==27:
                    break
            cap.release()
            cv2.destroyAllWindows()
                        


def InputPara(args,first): 
    if (first):     
        filepath='../emp'
        allFileName = file.eachFile(filepath)
        args.image_files=[];
        for x in allFileName: 
            if(os.path.getsize(x) < 10):
                print("image is too small ", x)
                continue
            args.image_files.append(x)
        
    args.model="../../model/20180402-114759/20180402-114759.pb"
        
    dict_1={"image_files":args.image_files,"image_size":args.image_size,"margin":args.margin,"gpu_memory_fraction":args.gpu_memory_fraction,"model":args.model,"detect_multiple_faces":args.detect_multiple_faces}
    return dict_1

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simhei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
          
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
    
 

    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)




if __name__ == '__main__':
    main()
 

    
