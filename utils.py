import math
import cv2
import numpy as np


def draw_points(img, pts1, pts2):
    """
    stereo omnidirectional image
    :param img:
    :param pts1:
    :param pts2:
    :return:
    """

    for i in range(pts1.shape[0]):
        point1 = pts1[i, :]
        ep1 = equirectangular(point1)
        ep1 = (ep1 + 1.0) * 128.0

        point2 = pts2[i, :]
        ep2 = equirectangular(point2)
        ep2 = (ep2 + 1.0) * 128.0
        ep2[1] += 256.0

        cv2.line(img, ep2.astype(np.uint16), ep1.astype(np.uint16), [0, 255, 255])
        cv2.drawMarker(img, ep1.astype(np.uint16), [255, 0, 0])
        cv2.drawMarker(img, ep2.astype(np.uint16), [0, 255, 0])

    cv2.line(img, (0, 256), (256, 256), (255, 255, 255))


def equirectangular(src):
    """
    transform 3d-point to 2d-equirectangular
    :param src: 3d point
    :return: 2d point on to equirectangular
    """
    longitude = math.atan2(src[2], src[0])
    latitude = math.atan2(src[1], math.sqrt(src[0] ** 2 + src[2] ** 2))

    x = longitude / math.pi
    y = (latitude * 2) / math.pi

    ret = np.zeros(2)
    ret[0] = x
    ret[1] = y
    return ret


def gen_points(cp):
    """
    generate random points (-10.0 to 10.0)
    :param cp: points quantity
    :return: points
    """
    # np.random.seed(seed=32)
    points = np.random.rand(cp, 3)
    points = (points - 0.5) * 20.0
    return points


def rotation_mat(rot):
    """
    generate rotation matrix
    :param rot:
    :return: rotation matrix
    """
    px = rot[0]
    py = rot[1]
    pz = rot[2]

    x = np.array([[1, 0, 0],
                  [0, np.cos(px), np.sin(px)],
                  [0, -np.sin(px), np.cos(px)]])
    y = np.array([[np.cos(py), 0, -np.sin(py)],
                  [0, 1, 0],
                  [np.sin(py), 0, np.cos(py)]])
    z = np.array([[np.cos(pz), np.sin(pz), 0],
                  [-np.sin(pz), np.cos(pz), 0],
                  [0, 0, 1]])
    return z @ y @ x


def transform(src, pos, rot):
    """
    transform point
    :param src:
    :param pos:
    :param rot:
    :return: transformed point
    """
    tmp = src.copy()

    rot_mat = rotation_mat(rot)
    tmp += pos
    tmp = tmp @ rot_mat

    return tmp


def projection(pos, rot, src):
    """
    project a points onto a unit-sphere
    :param pos: sphere center
    :param pos: sphere rotation
    :param src: source points
    :return: projected points
    """
    pts = src.copy()
    points = np.zeros((pts.shape[0], 3))  # output array
    for i in range(pts.shape[0]):
        tmp = transform(pts, pos, rot)
        points[i, :] = tmp[i, :] / np.linalg.norm(tmp[i, :])  # normalize
    return points
