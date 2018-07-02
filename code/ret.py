#!/usr/bin/python
# coding=utf-8
import sys, os
import cv2
import errno
#import numpy as np
#import pandas as pd

def data():
    img = []
    label = []
    for i in xrange(1,4):
        im_list = os.listdir('test1/out-test1_{}'.format(i))
        img.append(im_list)

    #for i in xrange(3):
    #    d = pd.read_csv('test1/txt/test1_{}_1.txt'.format(i), sep=";", header=None)
    #    label.append(d)
    #return label, img
    return img

def retify(img):
    try:
        os.makedirs('out')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
    fs = cv2.FileStorage('test1/calib/intrinsics.xml', cv2.FILE_STORAGE_READ)
    intrinsics = fs.getNode('floatdata').mat()
    fs.release()

    fs = cv2.FileStorage('test1/calib/distortion.xml', cv2.FILE_STORAGE_READ)
    distortion = fs.getNode('floatdata').mat()
    fs.release()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cv2.namedWindow('UNDISTORT')  # Create a named window
    cv2.moveWindow('UNDISTORT', 10, 10)
    #cv2.namedWindow(RAW_WIN)  # Create a named window
    #cv2.moveWindow(RAW_WIN, 650, 10)
    print 'Image Retification:'
    i = 3
    while(img != []):
        im = img[-1]
        while(im != []):
            frame = cv2.imread('test1/out-test1_{}/{}'.format(i,str(im[-1])))
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsics, distortion, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(intrinsics, distortion, None, newcameramtx, (w, h), 5)
            undist = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            cv2.imshow('UNDISTORT', undist)
            cv2.imwrite('out/{}'.format(im[-1]),undist)
            cv2.waitKey(3)
            del im[-1]
        del img[-1]
        i = i - 1
    print 'Retification Done!'

def main():
    img = data()
    retify(img)
main()
