import cv2
from PIL import Image
import numpy as np

from face_align import calculate_eyes
from utils import pts, draw_points

eye_distance = 80
cols, rows = 360, 640
target_eyes = ((cols * .5 + eye_distance, rows / 2),
               (cols * .5 - eye_distance, rows / 2))


img1 = cv2.imread("source/1.jpg")
eyes = calculate_eyes(img1)

M = cv2.getAffineTransform(pts(eyes), pts(target_eyes))
dst1 = cv2.warpAffine(img1, M, (cols, rows))

img2 = cv2.imread("source/2.jpg")
eyes = calculate_eyes(img2)

M = cv2.getAffineTransform(pts(eyes), pts(target_eyes))
dst2 = cv2.warpAffine(img2, M, (cols, rows))

imgs = [dst1, dst2]

for idx, img in enumerate(imgs):
    if idx == 0:
        first_img = img
    else:
        second_img = img
        second_weight = 1/(idx+1)
        first_weight = 1 - second_weight
        first_img = cv2.addWeighted(
            first_img, first_weight, second_img, second_weight, 0)

    cv2.imshow("destination", first_img)
    key = cv2.waitKey(3000)


# cv2.imshow("destination", first_img)


# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
