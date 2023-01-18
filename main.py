import glob
import math
import numpy as np
import cv2

import utils


def test():
    pc = 20

    rp = utils.gen_points(pc)
    pp = utils.projection([4, 2, 0], [0, np.radians(40.0), 0], rp)
    p2 = utils.projection([0, 0, -1], [0.0, 0.0, np.radians(40.0)], rp)

    run(pp, p2)


def find_points(img):
    print("--> find points")
    corners = utils.find_corners(img, (7, 10))
    pp = utils.equi_to_xyz_array(corners)
    print("--> end of process")

    color = utils.draw_point_on_img(img, corners)
    color = cv2.resize(color, dsize=(720, 480))
    cv2.imshow("img", color)
    cv2.waitKey(20)

    return pp


def stereo_img():
    upper_img_list = sorted(glob.glob("resources/Upper/*.JPG"))
    lower_img_list = sorted(glob.glob("resources/Lower/*.JPG"))

    if len(upper_img_list) != len(lower_img_list):
        print("Requires the same number of images from both cameras.")
        exit(-1)

    pp = np.empty([1, 3])
    pp2 = np.empty([1, 3])

    for i in range(len(upper_img_list)):
        img = cv2.imread(upper_img_list[i])
        img2 = cv2.imread(lower_img_list[i])

        print(upper_img_list[i])
        pp = np.append(pp, find_points(img), axis=0)
        print(lower_img_list[i])
        pp2 = np.append(pp2, find_points(img2), axis=0)

    pp = np.delete(pp, 0, 0)
    pp2 = np.delete(pp2, 0, 0)
    r, t = run(pp, pp2)

    t_cal_r = utils.direction_to_rotate(t)

    color = cv2.imread(upper_img_list[0])
    color = cv2.resize(color, (720, 480))
    map_x_1, map_y_1 = utils.rotate_equi((color.shape[1], color.shape[0]), t_cal_r)
    cal_img_1 = cv2.remap(color, map_x_1, map_y_1, cv2.INTER_LINEAR)
    cv2.imshow("calibrated_upper", cal_img_1)
    cv2.waitKey(0)

    color2 = cv2.imread(lower_img_list[0])
    color2 = cv2.resize(color2, (720, 480))
    map_x, map_y = utils.rotate_equi((color2.shape[1], color2.shape[0]), t_cal_r @ r)
    cal_img = cv2.remap(color2, map_x, map_y, cv2.INTER_LINEAR)
    cv2.imshow("calibrated_lower", cal_img)
    cv2.waitKey(0)

    np.savez("./cal_data", r, t)


def preview_points(title, pp, p2):
    img = np.zeros([480, 960, 3], dtype=np.uint8)
    utils.draw_points(img, pp, p2)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def run(pp, p2):
    preview_points('correspond', pp, p2)

    e = utils.find_essential_mat(pp, p2)
    r, t = utils.decompose_essential_mat(e, pp, p2)

    t_cal_r = utils.direction_to_rotate(t)

    pp_cal = np.zeros(pp.shape)
    p2_cal = np.zeros(p2.shape)
    for i in range(p2.shape[0]):
        pp_cal[i] = t_cal_r @ pp[i]
        p2_cal[i] = t_cal_r @ r @ p2[i]

    preview_points('correspond_calibrated', pp_cal, p2_cal)

    return r, t


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    stereo_img()
