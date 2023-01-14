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

    width = img.shape[1]
    height = img.shape[0]
    br = height / 2
    for i in range(pts1.shape[0]):
        point1 = pts1[i, :]
        ep1 = equirectangular(point1) + 1.0
        ep1[0] *= width / 2
        ep1[1] *= br / 2

        point2 = pts2[i, :]
        ep2 = equirectangular(point2) + 1.0
        ep2[0] *= width / 2
        ep2[1] *= br / 2
        ep2[1] += br

        cv2.line(img, ep2.astype(np.uint16), ep1.astype(np.uint16), [0, 255, 255])
        cv2.drawMarker(img, ep1.astype(np.uint16), [255, 0, 0])
        cv2.drawMarker(img, ep2.astype(np.uint16), [0, 255, 0])

    cv2.line(img, (0, int(br)), (width, int(br)), (255, 255, 255))


def equirectangular(src):
    """
    transform 3d-point to 2d-equirectangular
    :param src: 3d point
    :return: 2d point on to equirectangular
    """
    longitude = math.atan2(src[1], src[0])
    latitude = math.atan2(src[2], math.sqrt(src[0] ** 2 + src[1] ** 2))

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
    np.random.seed(seed=32)
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
    :param rot: sphere rotation
    :param src: source points
    :return: projected points
    """
    pts = src.copy()
    points = np.zeros((pts.shape[0], 3))  # output array
    for i in range(pts.shape[0]):
        tmp = transform(pts, pos, rot)
        points[i, :] = tmp[i, :] / np.linalg.norm(tmp[i, :])  # normalize
    return points


def skew_symmetric(src):
    return np.array([[0, -src[2], src[1]],
                     [src[2], 0, -src[0]],
                     [-src[1], src[0], 0]])


def find_corners(img, pattern_size):
    ret, corner = cv2.findChessboardCorners(img, pattern_size)

    pts = np.zeros((corner.shape[0], corner.shape[2]))
    pts[:, 0] = corner[:, 0, 0]
    pts[:, 1] = corner[:, 0, 1]

    corner = normalize_point(img, pts)
    return corner


def draw_point_on_img(img, corners):
    xg = img.shape[1] / 2
    yg = img.shape[0] / 2

    color = img.copy()
    for pt in corners:
        ip = (int((pt[0] + 1.0) * xg), int((pt[1] + 1.0) * yg))
        cv2.circle(color, ip, 20, [0, 0, 255], thickness=10)
    return color


def equi_to_xyz(src):
    longitude = src[0] * math.pi
    latitude = (src[1] * math.pi) / 2.0
    x = math.cos(latitude) * math.cos(longitude)
    y = math.cos(latitude) * math.sin(longitude)
    z = math.sin(latitude)
    return np.array([x, y, z])


def normalize_point(img, pts):
    shape = img.shape
    npts = pts.copy()

    npts[:, 0] = npts[:, 0] / (shape[1] / 2) - 1.0
    npts[:, 1] = npts[:, 1] / (shape[0] / 2) - 1.0
    return npts


def equi_to_xyz_array(corners):
    pts = np.zeros((corners.shape[0], 3))
    for i in range(corners.shape[0]):
        pts[i] = equi_to_xyz(corners[i])

    return pts


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

    sst1 = skew_symmetric(t1)
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


def rotate_equi(img_size, rot):
    width = img_size[0]
    height = img_size[1]

    x, y = np.meshgrid(np.linspace(-1.0, 1.0, width),
                       np.linspace(-1.0, 1.0, height))

    longitude = x * math.pi
    latitude = (y * math.pi) / 2

    nx = np.cos(latitude) * np.cos(longitude)
    ny = np.cos(latitude) * np.sin(longitude)
    nz = np.sin(latitude)

    rotm = np.zeros([nx.shape[0], nx.shape[1], 3])
    mix = np.dstack([np.dstack([nx, ny]), nz])
    for i in range(height):
        for j in range(width):
            rotm[i, j, :] = rot @ mix[i, j, :]

    nxr = rotm[:, :, 0]
    nyr = rotm[:, :, 1]
    nzr = rotm[:, :, 2]

    longitude_r = np.arctan2(nyr, nxr)
    latitude_r = np.arctan2(nzr, np.sqrt(nxr * nxr + nyr * nyr))
    ex = longitude_r / math.pi
    ey = 2 * latitude_r / math.pi

    ex = (ex + 1.0) * (width / 2.0)
    ey = (ey + 1.0) * (height / 2.0)

    return ex.astype(np.float32), ey.astype(np.float32)


def direction_to_rotate(direction):
    xaxis = np.cross(direction, np.array([0, 1, 0]))
    yaxis = np.cross(direction, xaxis)

    r = np.array([
        [xaxis[0], xaxis[1], xaxis[2]],
        [yaxis[0], yaxis[1], yaxis[2]],
        [direction[0], direction[1], direction[2]]
    ])

    # return utils.rotation_mat([0, np.radians(90.0), np.radians(180.0)]) @ r
    return r


def load_rt(path):
    print("load calibration data")
    cal = np.load(path)
    r = cal["arr_0"]
    t = cal["arr_1"]
    return r, t