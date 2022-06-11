import os
import numpy as np
from matplotlib import pyplot as plt
import pyransac3d as pyrsc
from pyransac3d import rodrigues_rot


def load_face_vectors():
    faces = list()
    face_vectors_files = os.listdir("faces")
    for file in face_vectors_files:
        if file.endswith(".npy"):
            try:
                face = np.load(f"faces/{file}")
                faces.append(face)
            except:
                print("bad face file")
                continue

    return faces


if __name__ == '__main__':
    vectors = np.asarray(load_face_vectors())
    # print(vectors)
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, vector in enumerate(vectors):
        center_point = [0, 0, 0]
        for point in vector:
            center_point += point / len(vector)
        # vector -= center_point
        point = pyrsc.Point()
        center, inliers = point.fit(vector, 10000)
        # print(center, inliers)
        # ax.scatter(center[0], center[1], center[2])
        vector = rodrigues_rot(vector, center_point, [0, 0, 1])
        # for j in range(3):
        #     vector[:, j] *= best_eq[j]
        #     vector[:, j] += best_eq[3]
        vector[:,2] -= 1
        ax.scatter(vector[:, 0], vector[:, 1], vector[:, 2])

    plt.show()
