import cv2
from face_combiner import combine_images

arr = ["imgs/example/1.jpg", "imgs/example/2.jpg"]
combine_images(arr, "imgs/main.png")

cv2.waitKey(0)

cv2.destroyAllWindows()
