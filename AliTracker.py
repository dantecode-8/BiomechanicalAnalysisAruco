import cv2 as aruco

# Open the device at the ID 0

cap = cv2.VideoCapture('watsapvideo1mark.mp4')
if not (cap.isOpened()):
    print("Could not open video device")

out = cv2.VideoWriter('c:\\outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 1)

cv2.namedWindow("TEST", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("TEST2", cv2.WINDOW_AUTOSIZE)
cv2.startWindowThread()

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
print(parameters)
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 400

marker = aruco.drawMarker(aruco_dict, 200, 200)
marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

x = 0
while (True):

    # Capture frame-by-frame

    ret, frame = cap.read()

    """
    grey2 = cv2.resize(frame2,(100,100))
    x_offset=10
    y_offset=10
    frame[y_offset:y_offset+grey2.shape[0], x_offset:x_offset+grey2.shape[1]] = cv2.cvtColor(grey2,cv2.COLOR_GRAY2BGR)
    """

    """
    x_offset=100
    y_offset=200
    frame[y_offset:y_offset+marker.shape[0], x_offset:x_offset+marker.shape[1]] = marker
    """
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame2, (10, 10), (100, 100), color=(255, 0, 0), thickness=3)

    corners, ids, rejected = aruco.detectMarkers(frame2, aruco_dict, parameters=parameters)

    for (i, b) in enumerate(corners):
        print(i, b, ids[i])
        print("B0", b[0])
        print("B00", b[0][0])
        print("B01", b[0][1])
        print("B02", b[0][2])
        c1 = (b[0][0][0], b[0][0][1])
        c2 = (b[0][1][0], b[0][1][1])
        c3 = (b[0][2][0], b[0][2][1])
        c4 = (b[0][3][0], b[0][3][1])
        print(c1)
        cv2.line(frame, c1, c2, (0, 0, 255), 3)
        cv2.line(frame, c2, c3, (0, 0, 255), 3)
        cv2.line(frame, c3, c4, (0, 0, 255), 3)
        cv2.line(frame, c4, c1, (0, 0, 255), 3)
        x = int((c1[0] + c2[0] + c3[0] + c4[0]) / 4)
        y = int((c1[1] + c2[1] + c3[1] + c4[1]) / 4)
        frame = cv2.putText(frame, str(ids[i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, "{0},{1}".format(int(c1[0]), int(c1[1])), (c1[0], c1[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, "{0},{1}".format(int(c2[0]), int(c2[1])), (c2[0], c2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, "{0},{1}".format(int(c3[0]), int(c3[1])), (c3[0], c3[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)
        frame = cv2.putText(frame, "{0},{1}".format(int(c4[0]), int(c4[1])), (c4[0], c4[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # if ids is not None: print (corners,ids)

    cv2.imshow('TEST', frame)
    cv2.imshow('TEST2', frame2)
    out.write(frame)

    c = cv2.waitKey(10)
    if (c == ord('q')):
        break
    # ret, frame = cap2.read()
    # cv2.imshow('TEST',frame)

    # Waits for a user input to quit the application
    c = cv2.waitKey(10)
    print(x, c)
    if (c == ord('q')):
        break

    x += 1

# When everything done, release the capture

cap.release()
out.release()

cv2.destroyAllWindows()