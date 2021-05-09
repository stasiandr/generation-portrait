import math
import cv2
import numpy as np

face_detector = cv2.CascadeClassifier("nn/haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("nn/haarcascade_eye.xml")


def euclidean_distance(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


def calculate_eyes(img):
    img = img.copy()
    faces = face_detector.detectMultiScale(img, 1.3, 5)

    face_x, face_y, face_w, face_h = faces[0]

    img = img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eye_detector.detectMultiScale(img_gray, 1.4, 5)

    color = (255, 0, 0)
    index = 0
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
        if index == 0:
            eye_1 = (eye_x, eye_y, eye_w, eye_h)
        elif index == 1:
            eye_2 = (eye_x, eye_y, eye_w, eye_h)
        else:
            color = (0, 0, 255)

        cv2.rectangle(img, (eye_x, eye_y),
                      (eye_x+eye_w, eye_y+eye_h), color, 2)
        index = index + 1

    if eye_1[0] > eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
    else:
        left_eye = eye_2
        right_eye = eye_1

    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)) + face_x,
                       int(left_eye[1] + (left_eye[3] / 2)) + face_y)

    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)) + face_x,
                        int(right_eye[1] + (right_eye[3]/2)) + face_y)

    return (left_eye_center, right_eye_center)
