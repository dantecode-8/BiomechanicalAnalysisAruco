import numpy as np
import cv2
import pickle
import glob
import cv2.aruco as aruco
import pandas as pd
import cv2
from PIL import Image

# Create arrays you'll use to store object points and image points from all images processed


objpoints = []  # 3D point in real world space where chess squares are
imgpoints = []  # 2D point in image plane, determined by CV2

# Chessboard variables
CHESSBOARD_CORNERS_ROWCOUNT = 9
CHESSBOARD_CORNERS_COLCOUNT = 6

# Theoretical object points for the chessboard we're calibrating against,
#
#     (CHESSBOARD_CORNERS_ROWCOUNT-1, CHESSBOARD_CORNERS_COLCOUNT-1, 0)
# Note that the Z value for all stays at 0, as this is a printed out 2D image
# And also that the max point is -1 of the max because we're zero-indexing
# The following line generates all the tuples needed at (0, 0, 0)
objp = np.zeros((CHESSBOARD_CORNERS_ROWCOUNT * CHESSBOARD_CORNERS_COLCOUNT, 3), np.float32)
# The following line fills the tuples just generated with their values (0, 0, 0), (1, 0, 0), ...
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_ROWCOUNT, 0:CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)

# Need a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-chessboard-<NUMBER>.jpg'
images = glob.glob('Chessboard[0-5].jpeg')
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
imageSize = None  # Determined at runtime

# Loop through images glob'ed
for iname in images:
    # Open the image
    img = cv2.imread(iname)
    # Grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard in the image, setting PatternSize(2nd arg) to a tuple of (#rows, #columns)
    board, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), None)

    # If a chessboard was found, let's collect image/corner points
    if board == True:
        # Add the points in 3D that we just discovered
        objpoints.append(objp)

        # Enhance corner accuracy with cornerSubPix
        corners_acc = cv2.cornerSubPix(
            image=gray,
            corners=corners,
            winSize=(9, 6),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                      0.001))  # Last parameter is about termination critera
        imgpoints.append(corners_acc)

        # If our image size is unknown, set it now
        if not imageSize:
            imageSize = gray.shape[::-1]

        # Draw the corners to a new image to show whoever is performing the calibration
        # that the board was properly detected
        img = cv2.drawChessboardCorners(img, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), corners_acc,
                                        board)
        # Pause to display each image, waiting for key press
        cv2.imshow('Chessboard', img)
        cv2.waitKey(0)
        cv2.imwrite('chessboardread1.jpeg', img)
    else:
        print("Not able to detect a chessboard in image: {}".format(iname))

# Destroy any open CV windows
cv2.destroyAllWindows()

# Make sure at least one image was found
if len(images) < 1:
    # Calibration failed because there were no images, warn the user
    print(
        "Calibration was unsuccessful. No images of chessboards were found. Add images of chessboards and use or alter the naming conventions used in this file.")
    # Exit for failure
    exit()

# Make sure we were able to calibrate on at least one chessboard by checking
# if we ever determined the image size
if not imageSize:
    # Calibration failed because we didn't see any chessboards of the PatternSize used
    print(
        "Calibration was unsuccessful. We couldn't detect chessboards in any of the images supplied. Try changing the patternSize passed into findChessboardCorners(), or try different pictures of chessboards.")
    # Exit for failure
    exit()

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=imageSize,
    cameraMatrix=None,
    distCoeffs=None)
matrix_coefficients = cameraMatrix
distortion_coefficients = distCoeffs

# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)
markersize = 5
#####################################################
inverservec = 0
inversetvec = 0
countframe = 0

foto = cv2.imread('fourmarkergood.jpeg')
frame_df = pd.DataFrame(columns = ['Frame_NUMBER','markerID','Sampling_time','Top_left_corner','Top_right','Bottom_right','Bottom_left'])
gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
parameters.adaptiveThreshConstant = 10
corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                            parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)
rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)
for (i, b) in enumerate(corners):
  c1 = (b[0][0][0], b[0][0][1])
  c2 = (b[0][1][0], b[0][1][1])
  c3 = (b[0][2][0], b[0][2][1])
  c4 = (b[0][3][0], b[0][3][1])
  centerX = (b[0][0][0] + b[0][1][0] + b[0][2][0] + b[0][3][0]) / 4
  centerY = (b[0][0][1] + b[0][1][1] + b[0][2][1] + b[0][3][1]) / 4
  center = (int(centerX), int(centerY))
  new_row = {'Frame_NUMBER':countframe ,'markerID': ids[i],'Sampling_time': 0 ,'Top_left_corner': c1,'Top_right': c2,'Bottom_right': c3,'Bottom_left':c4}
# append row to the dataframe
  frame_df = frame_df.append(new_row, ignore_index=True)
  frame_df.to_clipboard(sep='\t')

for i in range(0, ids.size):
# draw axis for the aruco markers
# Code to store Data in Data frame using pandas
# marker_info = pd.DataFrame({'id': [id(i)], 'corners': [corners[i][0]]}, columns=['id', 'corners'])
    aruco.drawAxis(foto, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], 0.06)

        # draw a square around the markers
    aruco.drawDetectedMarkers(foto, corners)

        # code to show ids of the marker found
    strg = ''
    for i in range(0, ids.size):
        strg += str(ids[i][0]) + ', '
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(foto, "Id: " + strg, (0, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)



    # display the resulting frame
    cv2.imshow('frame', foto)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

# When everything done, release the capture
foto.release()
cv2.destroyAllWindows()