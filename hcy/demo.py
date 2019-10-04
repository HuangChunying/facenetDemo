# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:34:30 2019

@author: train
"""

import facenet
def main():
    dataset = facenet.get_dataset("./input")
    print (dataset)

if __name__ == '__main__':
    main()
 