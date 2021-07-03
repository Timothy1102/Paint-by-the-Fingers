import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[7]

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.65, maxHands=1)
while True:
    success, img = cap.read()
    # resize the frame to fit the top bar
    img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList[0]) != 0 :
        print(len(lmList[0]))
        print(lmList[0][8][1:])
    else:
        print(lmList)

    # Setting the header image
    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break
