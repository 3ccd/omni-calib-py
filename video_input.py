import glob
import math
import numpy as np
import cv2

import utils
import stereoSGM


def fisheye_to_equi(map_xl, map_yl, map_xr, map_yr, rot_map_x, rot_map_y, equi):
    out1 = cv2.remap(equi[0:640, 0:640], map_xl, map_yl, cv2.INTER_LINEAR)
    out2 = cv2.remap(equi[0:640, 640:], map_xr, map_yr, cv2.INTER_LINEAR)
    out1[:, :320] = out2[:, :320]
    out1 = cv2.remap(out1, rot_map_x, rot_map_y, cv2.INTER_LINEAR)

    return out1


aperture = 202
print("gen map")
map_x_l, map_y_l = utils.fisheye_to_equi((480, 640), 640, aperture * (math.pi / 180))
map_x_r, map_y_r = utils.fisheye_to_equi((480, 640), 640, aperture * (math.pi / 180), False)
map_x, map_y = utils.rotate_equi((640, 480), utils.rotation_mat((0.0, np.radians(-90.0), 0.0)))
print("end")

cap1 = cv2.VideoCapture(1)
if not cap1.isOpened():
    exit(-2)

cap2 = cv2.VideoCapture(2)
if not cap2.isOpened():
    exit(-2)

while True:
    ret, img = cap1.read()
    ret, img1 = cap2.read()

    upper = fisheye_to_equi(map_x_l, map_y_l, map_x_r, map_y_r, map_x, map_y, img)
    lower = fisheye_to_equi(map_x_l, map_y_l, map_x_r, map_y_r, map_x, map_y, img1)
    upper = cv2.rotate(upper, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lower = cv2.rotate(lower, cv2.ROTATE_90_COUNTERCLOCKWISE)
    upper = cv2.GaussianBlur(upper, (5, 5), 3)
    lower = cv2.GaussianBlur(lower, (5, 5), 3)

    disparity = stereoSGM.run(upper, lower)

    cv2.imshow("disp", disparity)
    cv2.imshow("upper", upper)
    cv2.imshow("lower", lower)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

