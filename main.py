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
    print("find points")
    corners = utils.find_corners(img, (7, 10))
    pp = utils.equi_to_xyz_array(corners)

    color = utils.draw_point_on_img(img, corners)
    color = cv2.resize(color, dsize=(720, 480))
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

        print(upper_img_list[i])
        pp = np.append(pp, find_points(img), axis=0)
        print(lower_img_list[i])
        pp2 = np.append(pp2, find_points(img2), axis=0)

    pp = np.delete(pp, 0, 0)
    pp2 = np.delete(pp2, 0, 0)
    r, t = run(pp, pp2)

    t_cal_r = direction_to_rotate(t)

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


def run(pp, p2):
    np.set_printoptions(suppress=True)

    img = np.zeros([1024, 512, 3], dtype=np.uint8)

    utils.draw_points(img, pp, p2)
    cv2.imshow('show', img)
    cv2.waitKey(0)

    e = utils.find_essential_mat(pp, p2)
    r, t = utils.decompose_essential_mat(e, pp, p2)
    print(t)
    print(r)

    p2_cal = np.zeros(p2.shape)
    pp_cal = np.zeros(pp.shape)

    #t_cal_r = utils.rotation_mat(direction_to_euler(t))
    #print(t_cal_r)
    #t_cal_r = np.identity(3)
    t_cal_r = direction_to_rotate(t)
    print(t_cal_r)
    for i in range(p2.shape[0]):
        p2_cal[i] = t_cal_r @ r @ p2[i]
        pp_cal[i] = t_cal_r @ pp[i]

    img = np.zeros([1024, 512, 3], dtype=np.uint8)
    utils.draw_points(img, pp_cal, p2_cal)
    cv2.imshow('show1', img)
    cv2.waitKey(0)

    return r, t


def direction_to_rotate(direction):
    xaxis = np.cross(direction, np.array([0, 1, 0]))
    yaxis = np.cross(direction, xaxis)

    r = np.array([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [direction[0], direction[1], direction[2]]
    ])

    return utils.rotation_mat([0, np.radians(90.0), np.radians(180.0)]) @ r


def direction_to_euler(direction):
    pitch = math.atan2(direction[2], math.sqrt(direction[0] ** 2 + direction[1] ** 2))
    yaw = math.atan2(direction[1], direction[0])
    roll = math.acos(math.sqrt(direction[0] ** 2 + direction[1] ** 2) / math.sqrt(
        direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2))
    return pitch, roll, yaw


if __name__ == "__main__":
    stereo_img()
