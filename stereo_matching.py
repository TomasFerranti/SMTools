import cv2 as cv
import numpy as np
import sys
import copy

from shared_functions import saveToFile, readJson, readImage, plotCalibSegs


def sift(img1_BGR, img2_BGR):
    # SOURCE: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html

    img1_GRAY = cv.cvtColor(img1_BGR, cv.COLOR_BGR2GRAY)
    img2_GRAY = cv.cvtColor(img2_BGR, cv.COLOR_BGR2GRAY)

    # So first we need to find as many possible matches between two images to find the best translation.
    # For this, we use SIFT descriptors with FLANN based matcher and ratio test.
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1_GRAY, None)
    kp2, des2 = sift.detectAndCompute(img2_GRAY, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Now we have the list of best matches from both the images. Let's find the mean translation
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def getStereoFilename(filename):
    filename_split = filename.split(sep='_')

    swap = {'left': 'right', 'right': 'left'}
    filename_split[-1] = swap[filename_split[-1]]

    return '_'.join(filename_split)


def edgeMatch(img1_pts_match, img2_pts_match, edge):
    for point_idx in range(edge.shape[0]):
        cur_point = edge[point_idx, :]

        distances = np.sum((img1_pts_match - cur_point) ** 2, axis=1)
        distances = np.concatenate([distances.reshape(-1, 1),
                                    np.arange(distances.shape[0]).reshape(-1, 1)], axis=1)
        distances = distances[distances[:, 0].argsort()]

        knn_k = min(100, distances.shape[0])
        closest_points_indexes = distances[0:knn_k, 1]

        translation_diffs = img2_pts_match[closest_points_indexes,
                                           :] - img1_pts_match[closest_points_indexes, :]
        closest_ds = np.mean(translation_diffs, axis=0)

        edge[point_idx, :] = edge[point_idx, :] + closest_ds
    return edge


def stereoEdgesMatching(img1_data, img1_dict, img2_data, img2_dict):
    img1_pts_match, img2_pts_match = sift(img1_dict['img'], img2_dict['img'])

    cEscala, wInicio, hInicio = img2_dict['cEscala'], img2_dict['wInicio'], img2_dict['hInicio']
    for i in range(3):
        # scale to original image size
        img2_data['pontosguia'][i] = [[int((1 / cEscala) * (x - wInicio)),
                                       int((1 / cEscala) * (y - hInicio))]
                                      for x, y in img2_data['pontosguia'][i]]
        # input is in format list
        edges = [[img2_data['pontosguia'][i][2*j],
                  img2_data['pontosguia'][i][2*j + 1]]
                 for j in range(int(len(img2_data['pontosguia'][i]) / 2))]
        # match each edge on the other image
        edges = [edgeMatch(img1_pts_match, img2_pts_match, np.array(edge))
                 for edge in edges]
        # conver to list
        edges = [edge[point_idx, :].tolist()
                 for edge in edges for point_idx in range(edge.shape[0])]
        # scale back to calibration size
        edges = [[int((cEscala * x) + wInicio),
                  int((cEscala * y) + hInicio)]
                 for x, y in edges]

        img2_data['pontosguia'][i] = edges
    return img2_data


def main():
    filename = sys.argv[1]
    img1_data = readJson(filename)
    img1_dict = readImage(img1_data)
    img2_data = copy.deepcopy(img1_data)
    img2_data['nomeImagem'] = getStereoFilename(img1_data['nomeImagem'])
    img2_dict = readImage(img2_data)

    img2_data = stereoEdgesMatching(img1_data, img1_dict, img2_data, img2_dict)

    filepath_save = "cab_stereo_IMS/" + img2_data['nomeImagem'] + ".json"
    saveToFile(img2_data, filepath_save)


if __name__ == "__main__":
    main()
