
import numpy as np
import cv2

import utils


def test():
    pc = 20

    rp = utils.gen_points(pc)
    pp = utils.projection([0, 0, 0], [0, 0, 0], rp)
    p2 = utils.projection([0, 0, -1], [np.radians(10.0), np.radians(40.0), 0], rp)

    run(pp, p2)


def stereo_img():
    img = cv2.imread("resources/R0010146.JPG")
    img2 = cv2.imread("resources/R0010147.JPG")

    corners = utils.find_corners(img, (12, 8))
    print(corners)
    pp = utils.equi_to_xyz_array(corners)

    corners2 = utils.find_corners(img2, (12, 8))
    pp2 = utils.equi_to_xyz_array(corners2)

    color = utils.draw_point_on_img(img, corners)
    color = cv2.resize(color, dsize=(1280, 720))
    cv2.imshow("img", color)
    cv2.waitKey(0)

    color2 = utils.draw_point_on_img(img2, corners2)
    color2 = cv2.resize(color2, dsize=(1280, 720))
    cv2.imshow("img1", color2)
    cv2.waitKey(0)

    r, t = run(pp, pp2)


def run(pp, p2):
    np.set_printoptions(suppress=True)

    img = np.zeros([512, 256, 3], dtype=np.uint8)

    utils.draw_points(img, pp, p2)
    cv2.imshow('show', img)
    cv2.waitKey(0)

    e = utils.find_essential_mat(pp, p2)
    r, t = utils.decompose_essential_mat(e, pp, p2)

    p2_cal = np.zeros(p2.shape)
    for i in range(p2.shape[0]):
        p2_cal[i] = p2[i] @ r.transpose()

    img = np.zeros([512, 256, 3], dtype=np.uint8)
    utils.draw_points(img, pp, p2_cal)
    cv2.imshow('show1', img)
    cv2.waitKey(0)

    return r, t


if __name__ == "__main__":
    stereo_img()
