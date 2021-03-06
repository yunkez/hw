import glob
import numpy as np
import os
import math
from mpl_toolkits.mplot3d import Axes3D

from keras.utils import to_categorical

from tqdm import tqdm

import matplotlib.pyplot as plt

def read_pdb(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


def read_pdb_test(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()
        # print(stripped_line)

        splitted_line = stripped_line.split('\t')

        X_list.append(float(splitted_line[0]))
        Y_list.append(float(splitted_line[1]))
        Z_list.append(float(splitted_line[2]))
        atomtype_list.append(str(splitted_line[3]))

    return X_list, Y_list, Z_list, atomtype_list

def pro_lig_reader_full(grid_size=24, num_channels=4, grid_resolution=0.5,folder_name='training_data', test=False):

    print(">>> loading data...")
    pro_file_names = [os.path.basename(f) for f in sorted(glob.glob("./" + folder_name + "/*_pro_cg.pdb"))]
    lig_file_names = [os.path.basename(f) for f in sorted(glob.glob("./" + folder_name + "/*_lig_cg.pdb"))]

    # 3D grid with 0.5A resolution and 24A x 24A x 24A size
    # grid_size = grid_size
    # num_channels = num_channels
    # grid_resolution = grid_resolution
    grid_dim = math.floor(grid_size/grid_resolution)

    training_x = []
    training_y = []

    #for i in tqdm(range(len(pro_file_names))):
    for i in tqdm(range(25)):
        if test:
            pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb_test('./' + folder_name + '/' + pro_file_names[i][:4] + '_pro_cg.pdb')
        else:
            pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_file_names[i][:4] + '_pro_cg.pdb')
        pro = [pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list]

        #for j in range(len(lig_file_names)):
        for j in range(25):

            grid_3Dx4 = np.zeros((grid_dim, grid_dim, grid_dim, num_channels))
            if test:
                lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb_test('./' + folder_name + '/' + lig_file_names[j][:4] + '_lig_cg.pdb')
            else:
                lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_file_names[j][:4] + '_lig_cg.pdb')
            lig = [lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list]

            centroid = np.mean(lig[:3], axis=1)
            centroid = np.reshape(centroid, (3, 1))
            pro[:3] = np.add(np.subtract(pro[:3], centroid), grid_size/2)
            lig[:3] = np.add(np.subtract(lig[:3], centroid), grid_size/2)

            for atom in zip(pro[0], pro[1], pro[2], pro[3]):
                out_of_bound = 0
                channel = 4
                for d in atom[:3]:
                    if 0 > d or d >= (grid_size-grid_resolution):
                        out_of_bound = 1
                        break
                if out_of_bound == 1:
                    continue

                if atom[-1] == 'h':
                    channel = 0
                if atom[-1] == 'p':
                    channel = 1
                grid_3Dx4[math.floor(atom[0]/grid_resolution), math.floor(atom[1]/grid_resolution), math.floor(atom[2]/grid_resolution), channel] = 1

            for atom in zip(lig[0], lig[1], lig[2], lig[3]):
                out_of_bound = 0
                channel = 4
                for d in atom[:3]:
                    if 0 > d or d >= (grid_size-grid_resolution):
                        out_of_bound = 1
                        break
                if out_of_bound == 1:
                    continue

                if atom[-1] == 'h':
                    channel = 2
                if atom[-1] == 'p':
                    channel = 3
                grid_3Dx4[math.floor(atom[0]/grid_resolution), math.floor(atom[1]/grid_resolution), math.floor(atom[2]/grid_resolution), channel] = 1

            training_x.append(grid_3Dx4)
            if pro_file_names[i][:4] == lig_file_names[j][:4]:
                training_y.append(1)
            else:
                training_y.append(0)

            #plot_3D(grid_3Dx4)
    return np.stack(training_x, 0), to_categorical(training_y)


def pro_lig_reader_sample(pro_label='0001', lig_label='0001', grid_size=24, num_channels=4, grid_resolution=0.5, folder_name='training_data', test=False):

    # 3D grid with 0.5A resolution and 24A x 24A x 24A size
    grid_dim = math.floor(grid_size/grid_resolution)

    grid_3Dx4 = np.zeros((grid_dim, grid_dim, grid_dim, num_channels))

    if test:
        pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb_test('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
        lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb_test('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
    else:
        pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
        lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
    pro = [pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list]
    lig = [lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list]

    centroid = np.mean(lig[:3], axis=1)
    centroid = np.reshape(centroid, (3, 1))
    pro[:3] = np.add(np.subtract(pro[:3], centroid), grid_size/2)
    lig[:3] = np.add(np.subtract(lig[:3], centroid), grid_size/2)

    for atom in zip(pro[0], pro[1], pro[2], pro[3]):
        out_of_bound = 0
        channel = 4
        for d in atom[:3]:
            if 0 > d or d >= (grid_size-grid_resolution):
                out_of_bound = 1
                break
        if out_of_bound == 1:
            continue

        if atom[-1] == 'h':
            channel = 0
        if atom[-1] == 'p':
            channel = 1
        grid_3Dx4[math.floor(atom[0]/grid_resolution), math.floor(atom[1]/grid_resolution),
                  math.floor(atom[2]/grid_resolution), channel] = 1

    for atom in zip(lig[0], lig[1], lig[2], lig[3]):
        out_of_bound = 0
        channel = 4
        for d in atom[:3]:
            if 0 > d or d >= (grid_size-grid_resolution):
                out_of_bound = 1
                break
        if out_of_bound == 1:
            continue

        if atom[-1] == 'h':
            channel = 2
        if atom[-1] == 'p':
            channel = 3
        grid_3Dx4[math.floor(atom[0]/grid_resolution), math.floor(atom[1]/grid_resolution),
                  math.floor(atom[2]/grid_resolution), channel] = 1

    #plot_3D(grid_3Dx4)

    if pro_label == lig_label:
        return grid_3Dx4, 1
    else:
        return grid_3Dx4, 0

def plot_3D(grid_3Dx4):
    #fig = plt.figure()
    ax = plt.axes(projection='3d')

    for c, m, grid_3d in zip(['r', 'r', 'b', 'b'], ['o', '^', 'o', '^'], [grid_3Dx4[:, :, :, 0], grid_3Dx4[:, :, :, 1], grid_3Dx4[:, :, :, 2], grid_3Dx4[:, :, :, 3]]):
        xs = []
        ys = []
        zs = []
        for i in range(0, 48):
            for j in range(0, 48):
                for k in range(0, 48):
                    if grid_3d[i, j, k] == 1:
                        xs.append(i)
                        ys.append(j)
                        zs.append(k)
        ax.scatter(xs, ys, zs, c=c, marker=m)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

#pro_lig_reader_full()
#pro_lig_reader_sample()
