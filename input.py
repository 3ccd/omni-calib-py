import glob
import numpy as np
import cv2

import utils

map_x, map_y = utils.rotate_equi((5376, 2688), utils.rotation_mat((np.radians(-90.0), 0.0, 0.0)))


def rotate(files):
    for file in files:
        print("process for ", file)
        img = cv2.imread(file)
        cal_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        # tmp_img = cv2.resize(cal_img, (1280, 720))
        # cv2.imshow("calibrated", tmp_img)
        # cv2.waitKey(0)
        cv2.imwrite(file, cal_img)


if __name__ == "__main__":
    #rotate(glob.glob("resources/Lower/*.JPG"))
    #rotate(glob.glob("resources/Upper/*.JPG"))
    rotate(glob.glob("resources/input/*.JPG"))
