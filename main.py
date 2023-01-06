import glob
import numpy as np
import cv2

import utils


def test():
    pc = 20

    rp = utils.gen_points(pc)
    pp = utils.projection([0, 0, 0], [0, 0, 0], rp)
    p2 = utils.projection([0, 0, -1], [0.0, 0.0, np.radians(40.0)], rp)

    run(pp, p2)


def find_points(img):
    corners = utils.find_corners(img, (12, 8))
    pp = utils.equi_to_xyz_array(corners)

    color = utils.draw_point_on_img(img, corners)
    color = cv2.resize(color, dsize=(1280, 720))
    cv2.imshow("img", color)
    cv2.waitKey(0)

    return pp


def stereo_img():
    upper_img_list = sorted(glob.glob("resources/Upper/*"))
    lower_img_list = sorted(glob.glob("resources/Lower/*"))

    if len(upper_img_list) != len(lower_img_list):
        print("Requires the same number of images from both cameras.")
        exit(-1)

    pp = np.empty([1, 3])
    pp2 = np.empty([1, 3])

    for i in range(len(upper_img_list)):
        img = cv2.imread(upper_img_list[i])
        img2 = cv2.imread(lower_img_list[i])

        pp = np.append(pp, find_points(img), axis=0)
        pp2 = np.append(pp2, find_points(img2), axis=0)

    pp = np.delete(pp, 0, 0)
    pp2 = np.delete(pp2, 0, 0)
    r, _ = run(pp, pp2)

    color2 = cv2.imread(lower_img_list[0])
    color2 = cv2.resize(color2, (1280, 720))
    map_x, map_y = utils.rotate_equi((color2.shape[1], color2.shape[0]), r)
    cal_img = cv2.remap(color2, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow("calibrated", cal_img)
    cv2.waitKey(0)


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
