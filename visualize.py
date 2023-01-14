import open3d
import cv2
import numpy as np

import utils
import stereoSGM


def run(d_img):
    d_img = cv2.rotate(d_img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow("show", d_img)
    cv2.waitKey(0)
    pcd = open3d.geometry.PointCloud()

    for i in range(d_img.shape[0]):
        for j in range(d_img.shape[1]):
            ni = (i - (d_img.shape[0] / 2)) / d_img.shape[0] * 2
            ji = (j - (d_img.shape[1] / 2)) / d_img.shape[1] * 2
            pt = utils.equi_to_xyz([ji, ni])
            pt = pt * ((1 - d_img[i, j]) + 0.5) * 3

            if not d_img[i, j] <= 0.1:
                pcd.points.append(pt)

    open3d.visualization.draw_geometries(
        [pcd],
        width=600,
        height=600
    )



if __name__ == "__main__":
    run(np.load("parallax.npy"))
