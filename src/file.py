# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:27:21 2019

@author: train
"""

#import os
#
#fileName = []   
#
#def eachFile(filepath):
#    pathDir =  os.listdir(filepath)
#    for allDir in pathDir:
#        child = os.path.join('%s%s%s' % (filepath,"/", allDir))
#        if os.path.isfile(child):
#            #print(child)
#            fileName.append(child)
#            continue
#        eachFile(child)
#    res = fileName
#    fileName = []   
#    return res
    
# coding:utf-8
import os

def eachFile(path):
    all = []
    for fpathe,dirs,fs in os.walk(path):   # os.walk是获取所有的目录
        for f in fs:
            filename = os.path.join(fpathe,f)
            all.append(filename)
    return all

if __name__ == "__main__":
    b = eachFile('../tempImage')
    for i in b:
        print (i)
