# Distance Detector
# Author: Caio Jordão Carvalho

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    _, img = cam.read()

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Spliting image channels (i.e. Blue, Green and Red)
    b, g, r = cv2.split(img)

    # Subtracting grayscale from Red channel (calculate distance only from red objects)
    gray = cv2.subtract(r, gray)

    # Getting threshold
    _, gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # Eroding and dilating image
    gray = cv2.erode(gray, kernel, anchor=(-1, -1), iterations=4)
    gray = cv2.dilate(gray, kernel, anchor=(-1, -1), iterations=4)

    # Getting contours
    _, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    boundRect = []
    contor_poly = []

    # Getting contours rects
    for contour in contours:
        contor_poly.append(cv2.approxPolyDP(contour, 3, True))
        boundRect.append(cv2.boundingRect(contor_poly[len(contor_poly) - 1]))

    # Getting the rect with the maximum area
    max_index = 0
    max_area = 0

    for i in range(len(boundRect)):
        a = boundRect[i][1] * boundRect[i][2]

        cv2.rectangle(img, (boundRect[i][0], boundRect[i][1]), (boundRect[i][0] + boundRect[i][2], boundRect[i][1] + boundRect[i][3]), [255, 255, 0], 2, 8, 0)

        if a > max_area:
            max_area = a
            max_index = i

    distance = 0

    # Getting distance from the rect with the maximum area
    if len(boundRect) > 0:
        cv2.rectangle(img, (boundRect[max_index][0], boundRect[max_index][1]), \
                (boundRect[max_index][0]+boundRect[max_index][2], boundRect[max_index][1] + boundRect[max_index][3]), \
                [0, 255, 0], 2, 8, 0)
        # TODO: Fix this formula
        distance = 8414.7 * ((boundRect[max_index][2] * boundRect[max_index][3]) ** -0.468)


    h, w, _ = img.shape

    if distance > 0:
        cv2.putText(img, "%.2f cm." % distance, (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)
    else:
        cv2.putText(img, "Zero distance or couldn't calculate distance.", (w - 370, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

    cv2.imshow("Frame", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
