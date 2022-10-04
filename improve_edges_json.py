import improve_edge
import sys
import matplotlib.pyplot as plt
import copy

from shared_functions import saveToFile, readJson, readImage


def improveEdges(img_dict, data):
    data_improved = copy.deepcopy(data)
    edgesMatrix = improve_edge.cannyGaussian(img_dict['img'])
    cEscala, wInicio, hInicio = img_dict['cEscala'], img_dict['wInicio'], img_dict['hInicio']

    for i in range(0, 3):
        data_improved['pontosguia'][i] = [[int((1 / cEscala) * (x - wInicio)),
                                           int((1 / cEscala) * (y - hInicio))]
                                          for x, y in data_improved['pontosguia'][i]]

        edges = [[data_improved['pontosguia'][i][2*j],
                  data_improved['pontosguia'][i][2*j + 1]]
                 for j in range(int(len(data_improved['pontosguia'][i]) / 2))]

        best_edges = improve_edge.improveEdges(
            img_dict['img'], edges, plot=False, edgesMatrix=edgesMatrix)

        data_improved['pontosguia'][i] = [
            point for edge in best_edges for point in edge]
        data_improved['pontosguia'][i] = [[int(cEscala * x + wInicio), int(cEscala * y + hInicio)]
                                          for x, y in data_improved['pontosguia'][i]]
    return data_improved


def plotEdge(p0, p1, color, ax):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c=color)


def plotImprovement(img_dict, data, data_improved):
    img = img_dict["img_canvas"].copy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 20), dpi=80)
    axs[0].imshow(img)
    axs[1].imshow(img)
    for i, c in enumerate(['r', 'g', 'b']):
        edges = [[data['pontosguia'][i][2*j], data['pontosguia'][i][2*j + 1]]
                 for j in range(len(data['pontosguia'][i]) // 2)]
        edges_improved = [[data_improved['pontosguia'][i][2*j], data_improved['pontosguia'][i][2*j + 1]]
                          for j in range(len(data_improved['pontosguia'][i]) // 2)]
        for p0, p1 in edges:
            plotEdge(p0, p1, c, axs[0])
        for p0, p1 in edges_improved:
            plotEdge(p0, p1, c, axs[1])
    plt.show()


def improveJsonEdges(data):
    img_dict = readImage(data)
    data_improved = improveEdges(img_dict, data)
    plot = True
    if plot:
        plotImprovement(img_dict, data, data_improved)

    return data


def main():
    filename = sys.argv[1]
    data = readJson(filename)
    data = improveJsonEdges(data)
    filepath_save = "cab_stereo_IMS/" + data['nomeImagem'] + ".json"
    saveToFile(data, filepath_save)


if __name__ == "__main__":
    main()
