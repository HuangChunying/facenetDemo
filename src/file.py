# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:27:21 2019

@author: train
"""

import os


fileName=[]
def eachFile(filepath):
    
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s%s' % (filepath,"/", allDir))
        if os.path.isfile(child):
            #print(child)
            fileName.append(child)
            continue
        eachFile(child)
    return fileName
