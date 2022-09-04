import improve_edge
import json
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Carregar json
filePath = "..\\Produto1\\TextureExtractor\\public\\calib\\" + sys.argv[1] + " - Copia" + ".json"
with open(filePath, 'r') as file:
    data = json.load(file)

# Carregar e corrigir offset na imagem e arestas
imgPath = "..\\Produto1\\TextureExtractor\\public\\images\\" + data['nomeImagem'] + "." + data['extensao']
img = cv.imread(imgPath)
iWidth, iHeight = img.shape[1], img.shape[0]
cWidth, cHeight = 1200, 800
wInicio = 0
hInicio = 0 
cEscala = 0
aspectCanvas = cWidth/cHeight
aspectImg = iWidth/iHeight
if(aspectCanvas > aspectImg):
    cEscala = cHeight/iHeight
    wInicio = int(np.trunc((cWidth - cEscala*iWidth)/2))
else:
    cEscala = cWidth/iWidth
    hInicio = int(np.trunc((cHeight - cEscala*iHeight)/2))
img = cv.resize(img, (int(cEscala*iWidth), int(cEscala*iHeight)))

initialEdges = []
improvedEdges = []
# Melhorar as arestas guias
for i in range(0,3):
    data['pontosguia'][i] = [[x-wInicio, y-hInicio] for x,y in data['pontosguia'][i]]
    edges = [[data['pontosguia'][i][2*j], data['pontosguia'][i][2*j+1]] for j in range(int(len(data['pontosguia'][i])/2))]
    best_edges = improve_edge.improveEdges(img, edges, plot=False)
    data['pontosguia'][i] = [edge for edge_pair in best_edges for edge in edge_pair]
    data['pontosguia'][i] = [[int(x+wInicio), int(y+hInicio)] for x,y in data['pontosguia'][i]]
    initialEdges.append(edges)
    improvedEdges.append(best_edges)

def plotEdge(p0, p1, color, ax):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], c = color)

plot = True
if plot:
    fig, axs = plt.subplots(1, 2, figsize=(10,20), dpi=80)
    axs[0].imshow(img)
    axs[1].imshow(img)
    for i, c in enumerate(['r', 'g', 'b']):
        for p0, p1 in initialEdges[i]:
            plotEdge(p0, p1, c, axs[0])
        for p0, p1 in improvedEdges[i]:
            plotEdge(p0, p1, c, axs[1])
    plt.show()

# Salvar o resultado final
filePath = "C:\\Users\\Tomás Ferranti\\Desktop\\IMS\\Produto1\\TextureExtractor\\public\\calib\\" + sys.argv[1] + ".json"
save = True
if save:
    with open(filePath, 'w') as file:
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()