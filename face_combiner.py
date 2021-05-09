import cv2

from face_align import calculate_eyes
from utils import pts, draw_points

eye_distance = 80
cols, rows = 360, 640
target_eyes = ((cols * .5 + eye_distance, rows / 2),
               (cols * .5 - eye_distance, rows / 2))


def combine_images(img_paths, write_path=None):
    for idx, img_path in enumerate(img_paths):
        img = cv2.imread(img_path)
        eyes = calculate_eyes(img)

        M = cv2.getAffineTransform(pts(eyes), pts(target_eyes))
        img = cv2.warpAffine(img, M, (cols, rows))

        if idx == 0:
            first_img = img
        else:
            second_img = img
            second_weight = 1/(idx+1)
            first_weight = 1 - second_weight
            first_img = cv2.addWeighted(
                first_img, first_weight, second_img, second_weight, 0)

        cv2.imshow("destination", first_img)
        key = cv2.waitKey(1000)

        if (write_path != None):
            cv2.imwrite(write_path, first_img)
