"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""
import numpy as np
import cv2
import pickle
import glob
import cv2.aruco as aruco
import pandas as pd

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
            winSize=(11, 11),
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

###------------------ ARUCO TRACKER ---------------------------
###########
countframe = 0
frame_df = pd.DataFrame(columns = ['Frame_NUMBER','markerID','Sampling_time','Top_left_corner','Top_right','Bottom_right','Bottom_left'])
cap = cv2.VideoCapture('watsapvideo3markers.mp4')



while (True) :
    ret, frame = cap.read()
    countframe +=1

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                            parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)
    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, matrix_coefficients, distortion_coefficients)
    #rvec = np.transpose(rvec)
    #recabcorners, mrv = rotate_marker_corners(rvec,markersize,tvec)

    for (i, b) in enumerate(corners):
        print(i, b, ids[i])
        print("B0", b[0])
        print("B00", b[0][0])
        print("B01", b[0][1])
        print("B02", b[0][2])
        print(countframe)
        c1 = (b[0][0][0], b[0][0][1])
        c2 = (b[0][1][0], b[0][1][1])
        c3 = (b[0][2][0], b[0][2][1])
        c4 = (b[0][3][0], b[0][3][1])
        x_sum = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
        y_sum = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]

        x_centerPixel = x_sum * .25
        y_centerPixel = y_sum * .25
        new_row = {'Frame_NUMBER':countframe ,'markerID': ids[i],'Sampling_time': 0 ,'Top_left_corner': c1,'Top_right': c2,'Bottom_right': c3,'Bottom_left':c4}
        # append row to the dataframe
        frame_df = frame_df.append(new_row, ignore_index=True)
        print(c1)
        print(x_centerPixel)
        print(y_centerPixel)
        cv2.line(frame, c1, c2, (0, 0, 255), 3)
        cv2.line(frame, c2, c3, (0, 255, 0), 3)
        cv2.line(frame, c3, c4, (0, 0, 255), 3)
        cv2.line(frame, c4, c1, (0, 0, 255), 3)
        x = int((c1[0] + c2[0] + c3[0] + c4[0]) / 4)
        y = int((c1[1] + c2[1] + c3[1] + c4[1]) / 4)
        frame = cv2.putText(frame, str(ids[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_df.to_clipboard(sep='\t')

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error


        for i in range(0, ids.size):
            # draw axis for the aruco markers
            # Code to store Data in Data frame using pandas
            # marker_info = pd.DataFrame({'id': [id(i)], 'corners': [corners[i][0]]}, columns=['id', 'corners'])
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec[i], tvec[i], 0.04)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ', '

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

