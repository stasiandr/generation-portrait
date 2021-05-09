import numpy as np


def pts(my_eyes):
    additional_point = (my_eyes[1][0] + my_eyes[1][1] - my_eyes[0][1],
                        my_eyes[1][1] - my_eyes[1][0] + my_eyes[0][0])

    return np.float32([*my_eyes, additional_point])


def draw_points(img, points):
    for point in points:
        eye = tuple(map(int, point))
        img = cv2.circle(img, eye, 5, (255, 0, 0), -1)
