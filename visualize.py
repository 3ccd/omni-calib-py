import open3d
import cv2
import numpy as np

import utils
import stereoSGM


def run(d_img, color=None):
    d_img = cv2.rotate(d_img, cv2.ROTATE_90_CLOCKWISE)
    color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
    color = cv2.resize(color, [d_img.shape[1], d_img.shape[0]])
    cv2.imshow("show", d_img)
    cv2.waitKey(0)
    cv2.imshow("show1", color)
    cv2.waitKey(0)

    pcd = open3d.geometry.PointCloud()
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    for i in range(d_img.shape[0]):
        for j in range(d_img.shape[1]):
            ni = (i - (d_img.shape[0] / 2)) / d_img.shape[0] * 2
            ji = (j - (d_img.shape[1] / 2)) / d_img.shape[1] * 2
            vec = utils.equi_to_xyz([ji, ni])
            norm = (1 - d_img[i, j])
            pt = vec * (norm + (np.cos(ni)))**2

            if not d_img[i, j] <= 0.1:
                pcd.points.append([pt[0], pt[1], pt[2]])
                pcd.colors.append(color[i, j, :] / 255.0)

    open3d.visualization.draw_geometries(
        [pcd],
        width=600,
        height=600
    )



if __name__ == "__main__":
    color_img = cv2.imread("tmp/lower_tmp.png")
    run(np.load("parallax.npy"), color_img)
