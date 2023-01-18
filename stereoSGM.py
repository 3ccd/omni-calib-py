import numpy as np
import cv2

import glob
import os

import utils


def load_img(upper_img, lower_img):
    print("loading : ", upper_img)
    print("loading : ", lower_img)
    imgU = cv2.resize(cv2.imread(upper_img), [720, 720])
    imgL = cv2.resize(cv2.imread(lower_img), [720, 720])
    # imgU = cv2.imread(upper_img_list[0])
    # imgL = cv2.imread(lower_img_list[0])
    return imgU, imgL


def calc_map(img, r_mat):
    map_x, map_y = utils.rotate_equi((img.shape[1], img.shape[0]), r_mat)
    cal_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    cal_img = cv2.rotate(cal_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cal_img


def show(out):
    out[out < 0.0] = 0
    out = (out * 255).astype(np.uint8)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)

    out = cv2.rotate(cv2.resize(out, [720, 1280]), cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('disparity', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_calibrated_img(imgU, imgL, r, t):
    tmp_file_name_u = "tmp/upper_tmp.png"
    tmp_file_name_l = "tmp/lower_tmp.png"

    is_file = os.path.isfile(tmp_file_name_u)
    if is_file:
        cal_img_u = cv2.imread(tmp_file_name_u)
        cal_img_l = cv2.imread(tmp_file_name_l)
    else:
        t_cal_r = utils.direction_to_rotate(t)

        print("calc map for upper image")
        cal_img_u = calc_map(imgU, t_cal_r)
        cv2.imwrite(tmp_file_name_u, cal_img_u)

        print("calc map for lower image")
        cal_img_l = calc_map(imgL, t_cal_r @ r)
        cv2.imwrite(tmp_file_name_l, cal_img_l)

    return cal_img_u, cal_img_l


def run(cal_img_u, cal_img_l):

    # disparity range is tuned for 'aloe' image pair
    window_size = 9
    min_disp = 0
    num_disp = 16 * 5
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   P1=8 * 3 * window_size ** 2,
                                   # P1=window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=0,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

    print('computing disparity...')
    disparity = stereo.compute(cal_img_u, cal_img_l).astype(np.float32) / 16.0
    out = ((disparity - min_disp) / num_disp)
    out = out[:, num_disp:-num_disp]

    return out


if __name__ == "__main__":
    img_u, img_l = load_img("resources/input/R0010028.JPG", "resources/input/R0010187.JPG")
    r_, t_ = utils.load_rt("cal_data.npz")
    cal_u, cal_l = get_calibrated_img(img_u, img_l, r_, t_)
    ret = run(cal_u, cal_l)
    np.save("parallax", ret)
    show(ret)
