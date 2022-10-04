import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def saveToFile(data, filepath):
    with open(filepath, 'w') as file:
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()


def readJson(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def readImage(data, basepath="imagens_stereo_IMS/"):
    imgPath = basepath + data['nomeImagem'] + "." + data['extensao']
    img = cv.imread(imgPath)

    iWidth, iHeight = img.shape[1], img.shape[0]
    cWidth, cHeight = 1200, 800
    wInicio = 0
    hInicio = 0
    cEscala = 0
    aspectCanvas = cWidth/cHeight
    aspectImg = iWidth/iHeight

    if (aspectCanvas > aspectImg):
        cEscala = cHeight/iHeight
        wInicio = int(np.trunc((cWidth - cEscala*iWidth)/2))
    else:
        cEscala = cWidth/iWidth
        hInicio = int(np.trunc((cHeight - cEscala*iHeight)/2))
    img_canvas = cv.resize(img, (int(cEscala*iWidth), int(cEscala*iHeight)))

    if wInicio == 0:
        white_rect = 255 * \
            np.ones(shape=(hInicio, img_canvas.shape[1], img_canvas.shape[2])).astype(
                np.uint8)
        img_canvas = np.concatenate([white_rect, img_canvas], axis=0)
    else:
        white_rect = 255 * \
            np.ones(shape=(img_canvas.shape[0], wInicio, img_canvas.shape[2])).astype(
                np.uint8)
        img_canvas = np.concatenate([white_rect, img_canvas], axis=1)

    img_dict = {}
    img_dict['img'] = img
    img_dict['img_canvas'] = img_canvas
    img_dict['cEscala'] = cEscala
    img_dict['wInicio'] = wInicio
    img_dict['hInicio'] = hInicio
    return img_dict


def plotEdge(p0, p1, color, ax):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c=color)


def plotCalibSegs(img_dict, data):
    img = img_dict["img_canvas"].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 20), dpi=80)
    ax.imshow(img)
    for i, c in enumerate(['r', 'g', 'b']):
        edges = [[data['pontosguia'][i][2*j], data['pontosguia'][i][2*j + 1]]
                 for j in range(len(data['pontosguia'][i]) // 2)]
        for p0, p1 in edges:
            plotEdge(p0, p1, c, ax)
    plt.show()
