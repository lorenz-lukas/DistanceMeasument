#!/usr/bin/python
# coding=utf-8
import cv2
import numpy as np

def main():
    exit = False
    cv2.namedWindow('frame')  # Create a named window
    cv2.moveWindow('frame', 10, 10)
    Y = 130
    X = 340
    while(not exit):
        im = cv2.imread('test1/out-test1_1/1_13316.png')
        im = cv2.resize(im,(940,540) )#.astype(np.float32)
        cv2.rectangle(im,(X,Y),(X+224,Y+224),(255,255,0),7)
        cv2.imshow('frame',im)
        if cv2.waitKey(3)& 0xFF == ord('q'):
            exit = True
main()
