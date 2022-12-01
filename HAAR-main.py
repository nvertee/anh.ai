import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

test_cases = ['testTho.jpg','img_2.png','img_1.png']

img = cv2.imread(test_cases[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

for (x, y, w, h) in faces:
	roi_gray = gray[y:y + h, x:x + w]
	roi_color = img[y:y + h, x:x + w]
	mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 80)
	for (mx, my, mw, mh) in mouth:
		cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('test-result.jpg', img)
cv2.destroyAllWindows()
