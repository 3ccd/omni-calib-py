
import numpy as np
import cv2

import utils


def find_essential_mat(pts1, pts2):
    emat = np.zeros((pts1.shape[0], 9))

    for i in range(pts1.shape[0]):
        tmp = np.array([pts1[i, 0] * pts2[i, 0], pts1[i, 0] * pts2[i, 1], pts1[i, 0] * pts2[i, 2],
                        pts1[i, 1] * pts2[i, 0], pts1[i, 1] * pts2[i, 1], pts1[i, 1] * pts2[i, 2],
                        pts1[i, 2] * pts2[i, 0], pts1[i, 2] * pts2[i, 1], pts1[i, 2] * pts2[i, 2]])
        emat[i, :] = tmp

    u, s, vh = np.linalg.svd(emat)
    vec = vh[8, :].reshape(3, 3)

    ue, se, vhe = np.linalg.svd(vec)
    ret = ue @ np.diag([1, 1, 0]) @ vhe
    # print(ret)
    # print(emat @ ret.reshape(9, -1))

    return ret


def decompose_essential_mat(e, pts, pts2):
    eet = e @ e.transpose()
    u, s, vh = np.linalg.svd(eet)
    t1 = vh[2, :]
    print(s)

    sst1 = utils.skew_symmetric(t1)
    sum_ab = 0
    for i in range(pts.shape[0]):
        a = sst1 @ pts[i]
        b = e @ pts2[i]
        sum_ab += a.transpose() @ b

    t2 = -t1
    if sum_ab >= 0:
        e1 = e
    else:
        e1 = -e

    k = -sst1 @ e1
    uk, sk, vhk = np.linalg.svd(k)
    r = uk @ np.diag([1, 1, np.linalg.det(uk @ vhk)]) @ vhk
    print(r)

    sum_ab = 0
    for i in range(pts.shape[0]):
        a = np.cross(r.transpose() @ t1, r.transpose() @ pts[i])
        b = np.cross(r.transpose() @ pts[i], pts2[i])
        sum_ab += a.transpose() @ b

    if sum_ab > 0:
        t = t1
    else:
        t = t2

    return r, t


def run():
    np.set_printoptions(suppress=True)

    img = np.zeros([512, 256, 3], dtype=np.uint8)

    pc = 20

    rp = utils.gen_points(pc)
    pp = utils.projection([0, 0, 0], [0, 0, 0], rp)
    p2 = utils.projection([0, -1, 0], [np.radians(10.0), np.radians(40.0), 0], rp)
    utils.draw_points(img, pp, p2)
    cv2.imshow('show', img)
    cv2.waitKey(0)

    e = find_essential_mat(pp, p2)
    r, t = decompose_essential_mat(e, pp, p2)

    p2_cal = np.zeros(p2.shape)
    for i in range(p2.shape[0]):
        p2_cal[i] = p2[i] @ r.transpose()

    img = np.zeros([512, 256, 3], dtype=np.uint8)
    utils.draw_points(img, pp, p2_cal)
    cv2.imshow('show1', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    run()



