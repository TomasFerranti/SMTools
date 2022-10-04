import sys
import numpy as np

from shared_functions import saveToFile, readJson


def getCalibType(data):
    cab_type = "normal"
    missing_idx = None
    for dim in range(0, 3):
        if len(data['pontosguia'][(dim) % 3]) == 0 and len(data['pontosguia'][(dim + 1) % 3]) > 0 and len(data['pontosguia'][(dim + 2) % 3]) > 0:
            cab_type = "centrado"
            missing_idx = dim
            break
    return cab_type, missing_idx


def triangleArea(p, q, r):
    return p[0]*q[1] + q[0]*r[1] + r[0]*p[1] - p[1]*q[0] - q[1]*r[0] - r[1]*p[0]


def lineIntersection(edge1, edge2):
    p, q = edge1
    r, s = edge2
    a1 = triangleArea(p, q, r)
    a2 = triangleArea(q, p, s)
    amp = a1 / (a1 + a2)
    result = np.array(r) * (1 - amp) + np.array(s) * amp
    return result.tolist()


def getVanishingPoints(data, missing_idx):
    missing_idx_cases = {None: [0, 1, 2], 0: [1, 2], 1: [0, 2], 2: [0, 1]}
    dim_cases = missing_idx_cases[missing_idx]
    vanishing_points = []
    for dim in dim_cases:
        intersection_points = []
        edges = [[data['pontosguia'][dim][2*j], data['pontosguia'][dim][2*j+1]]
                 for j in range(len(data['pontosguia'][dim])//2)]
        for edge_index_1 in range(len(edges)):
            for edge_index_2 in range(edge_index_1 + 1, len(edges)):
                edge1 = edges[edge_index_1]
                edge2 = edges[edge_index_2]
                intersection_points.append(lineIntersection(edge1, edge2))
        intersection_points = [np.array(p).reshape(-1, 2)
                               for p in intersection_points]
        intersection_points = np.concatenate(intersection_points, axis=0)
        vanishing_point = np.mean(intersection_points, axis=0)
        vanishing_point = vanishing_point.tolist()
        vanishing_points.append(vanishing_point)
    data['pontosfuga'] = vanishing_points


def proj(Va, Vb, q):
    c = Va[0]*Vb[0] + Va[1]*Vb[1]
    v = Vb[0]*Vb[0] + Vb[1]*Vb[1]
    Vb = np.array(Vb)
    q = np.array(q)
    P = Vb * c/v
    return P + q


def addHom(arr):
    return np.array([arr[0], arr[1], 0])


def getOpticalCenter(data, missing_idx):
    if missing_idx == None:
        Fx, Fy, Fz = data['pontosfuga']
        Fx, Fy, Fz = np.array(Fx), np.array(Fy), np.array(Fz)
        hx = proj(Fx - Fy, Fz - Fy, Fy)
        hy = proj(Fy - Fz, Fx - Fz, Fz)
        CO = np.array(lineIntersection([Fx, hx], [Fy, hy]))
    else:
        Fx, Fy = data['pontosfuga']
        Fx, Fy = np.array(Fx), np.array(Fy)
        CO = np.array([1200 / 2, 800 / 2])
        n = Fy - Fx
        n = np.array([-n[1], n[0]])
        t_par = (np.linalg.norm(CO))**2 + Fx.dot(Fy) - CO.dot(Fx - Fy)
        t_par = t_par / (Fx.dot(n) - CO.dot(n))
        n = n * t_par
        Fz = CO + n
        vanishing_points = [Fx, Fy, Fz]
        cases = {0: [2, 0, 1], 1: [0, 2, 1], 2: [0, 1, 2]}
        case = cases[missing_idx]
        vanishing_points = [vanishing_points[case[dim]] for dim in range(0, 3)]
        [Fx, Fy, Fz] = vanishing_points
    z2 = ((Fx[0] - Fy[0])**2 + (Fx[1] - Fy[1])**2)
    z2 -= ((Fx[0] - CO[0])**2 + (Fx[1] - CO[1])**2)
    z2 -= ((Fy[0] - CO[0])**2 + (Fy[1] - CO[1])**2)
    z2 = -1 * np.sqrt(z2/2)
    C = np.array([CO[0], CO[1], z2])
    Fx, Fy, Fz = addHom(Fx), addHom(Fy), addHom(Fz)
    X, Y, Z = Fx - C, Fy - C, Fz - C
    X, Y, Z = X / np.linalg.norm(X), Y / \
        np.linalg.norm(Y), Z / np.linalg.norm(Z)
    baseXYZ = np.concatenate(
        [X.reshape(1, -1), Y.reshape(1, -1), Z.reshape(1, -1)], axis=1).ravel()
    data["base"] = baseXYZ.tolist()
    data["centrooptico"] = CO.tolist()
    data["camera"] = C.tolist()


def calibrateCamera(data):
    cab_type, missing_idx = getCalibType(data)
    getVanishingPoints(data, missing_idx)
    getOpticalCenter(data, missing_idx)
    return data


def main():
    filename = sys.argv[1]
    data = readJson(filename)
    data = calibrateCamera(data)
    filepath_save = "cab_stereo_IMS/" + data['nomeImagem'] + ".json"
    saveToFile(data, filepath_save)


if __name__ == "__main__":
    main()
