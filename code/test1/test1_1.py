#!/usr/bin/python
# coding=utf-8
import sys, os
import cv2
import numpy as np
import pandas as pd
import math
import csv
import errno

def readCSV():
    d = []
    for i in xrange(4):
        data = pd.read_csv('txt/test1_1_1.txt', sep=";", header=None)
        d.append(data)
    return d

def sync(data):
    t_frame = 33.333 #ms
    td = data[0]
    t = td[0] # time in ms
    d = td[1] # and distance in cm# time in ms
    cap = cv2.VideoCapture('video/test1_1_1.mp4')# 1920x1080 and 3030FPS
    j = 1                                      # One frame takes 0.033_ secs
    start = True
    frame_wait = 0
    ret = True
    print "Distance (cm):\n"
    cv2.namedWindow('cap')
    try:
        os.makedirs('out-test1_1')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
    while(ret and (j < (len(t)-1))):
        ret, frame = cap.read()
        if ret:
            if frame_wait > 0:
                frame_wait -= 1
            else: # sync with camera
                frame_wait = math.ceil((t[j]-t[j-1])/(1.4*t_frame))
                cv2.imwrite('out-test1_1/1_{}.png'.format(t[j]),frame)
                cv2.imshow('cap', frame)
                #print '1'
                print d[j]
                j+=1
            cv2.waitKey(3)
    print 'Sync done!'
    cap.release()
    cv2.destroyAllWindows()

def main():
    print '####################################################'
    print '##                  Opencv3.4.0                   ##'
    print '##                   python2.7                    ##'
    print '##       Sync Camera and Ultrasonic sensor        ##'
    print '####################################################'
    print '\n\n\n'
    print 'Please, wait a moment.'
    data = readCSV()
    sync(data)

main()
