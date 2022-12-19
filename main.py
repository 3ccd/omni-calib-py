
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


def decompose_essential_mat(e, pts):
    eet = e @ e.transpose()
    u, s, vh = np.linalg.svd(eet)
    t1 = vh[2, :]
    print(t1)

    sst1 = utils.skew_symmetric(t1)
    sum_ab = 0
    for pt in pts:
        a = sst1 @ pt
        b = e @ pt
        sum_ab += np.dot(a.transpose(), b)

    t2 = -t1
    if sum_ab >= 0:
        e1 = e
        e2 = -e
    else:
        e1 = -e
        e2 = e

    k = np.cross(-t1, e1)
    uk, sk, vhk = np.linalg.svd(k)


def run():
    np.set_printoptions(suppress=True)

    img = np.zeros([512, 256, 3], dtype=np.uint8)

    pc = 20

    rp = utils.gen_points(pc)
    pp = utils.projection([1, 2, 3], [0, 0, 0], rp)
    p2 = utils.projection([0, -3, 0], [np.radians(60), np.radians(20.0), 0], rp)
    utils.draw_points(img, pp, p2)
    cv2.imshow('show', img)
    cv2.waitKey(0)

    e = find_essential_mat(pp, p2)
    # r1, r2, t = cv2.decomposeEssentialMat(e)
    decompose_essential_mat(e, pp)


if __name__ == "__main__":
    run()



