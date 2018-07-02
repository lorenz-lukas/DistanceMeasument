#!/usr/bin/python3
# coding=utf-8
import sys, os
import cv2
import numpy as np
import pandas as pd
import math

SNAPSHOT_WIN = "Snapshot"
UNDISTORT_WIN = "Undistort"
RAW_WIN = "RAW Video"

CIRCLE_RADIUS = 5
CIRCLE_COLOR = (0,0,255)
RULLER_COLOR = (255,0,0)

raw_click_count = 0
raw_p1 = (0,0)
raw_p2 = (0,0)

und_click_count = 0
und_p1 = (0,0)
und_p2 = (0,0)



# Function that effectivaly to the project job
def calibrate_round():
    intrinsics_df = pd.DataFrame()
    distortion_df = pd.DataFrame()

    global click_count
    global p1, p2

    n_spanspots = 20
    spanspot_count = 0

    frame_step = 10
    frame_wait = 10

    board_w = 6;
    board_h = 8;
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_h * board_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

    start = True
    cap = cv2.VideoCapture('calib2.MP4')

    for round in range(7):
        print("\nCalibration round %d #################################" % (round+1))
        spanspot_count = 0
        while spanspot_count < n_spanspots:
            ret, frame = cap.read()

            if ret:

                if frame_wait > 0:
                    frame_wait -= 1
                else:
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    found, corners = cv2.findChessboardCorners(frame_gray, (board_w, board_h), None)
                    if start and found and (len(corners) == board_w * board_h):

                        frame_wait = frame_step

                        spanspot_count += 1
                        objpoints.append(objp)

                        corners2 = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners2)
                        frame = cv2.drawChessboardCorners(frame, (board_w, board_h), corners2, ret)
                        #cv2.imshow(SNAPSHOT_WIN, frame) # Se achou mostra snapshot colorido
                        if spanspot_count ==20:
                            break
                    else:
                        #cv2.imshow(SNAPSHOT_WIN, frame_gray) # Se nao achou mostra snapshot cinza
                        pass
                #cv2.imshow(RAW_WIN, frame)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_gray.shape[::-1],None, None)

        intrinsics = np.array(intrinsics).reshape(1,9)
        distortion = np.array(distortion).reshape(1,5)

        intrinsics_df = intrinsics_df.append(pd.DataFrame(data=intrinsics,
                                                          columns=['p1', 'p2', 'p3',
                                                                   'p4', 'p5', 'p6',
                                                                   'p7', 'p8', 'p9'])).reset_index(drop=True)
        distortion_df = distortion_df.append(pd.DataFrame(data=distortion,
                                                          columns=['p1', 'p2', 'p3', 'p4', 'p5'])).reset_index(drop=True)

    # When everything done, release the capture
    print 'Calibration done!!! Generating files.'
    cap.release()
    cv2.destroyAllWindows()
    return intrinsics_df, distortion_df

def calibrate():

    intrinsics_df, distortion_df = calibrate_round()

    intrinsics = np.array(intrinsics_df.mean()).reshape(3,3)
    distortion = np.array(distortion_df.mean()).reshape(1,5)

    intrinsics_std = np.array(intrinsics_df.std()).reshape(3,3)
    distortion_std = np.array(distortion_df.std()).reshape(1,5)

    print("\nFinal intrinsic parameters ###########################")
    print("mean:")
    print(intrinsics)
    print("std:")
    print(intrinsics_std)

    print("\nFinal distortion parameters #########################:")
    print("mean:")
    print(distortion)
    print("std:")
    print(distortion_std)

    print("\nWriting xml files #########################:")
    fs = cv2.FileStorage('intrinsics.xml', cv2.FILE_STORAGE_WRITE)
    fs.write("floatdata", intrinsics)
    fs.release()

    fs = cv2.FileStorage('distortion.xml', cv2.FILE_STORAGE_WRITE)
    fs.write("floatdata", distortion)
    fs.release()

def do_the_job():
    calibrate()

# Main Function
def calib():
    print '####################################################'
    print '##                  Opencv3.4.0                   ##'
    print '##                   python2.7                    ##'
    print '##                Camera Calibration              ##'
    print '####################################################'
    
    
    if len(sys.argv) != 1:
        print('\nSyntax: %s\n' % sys.argv[0])
        sys.exit(-1)

    do_the_job()

if __name__ == '__main__':
  calib()
