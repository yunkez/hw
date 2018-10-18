import glob
import numpy as np
import os
import math
from read_pdb_file import *


def cal_distance_by_label(pro_label='0001', lig_label='0001', folder_name='training_data'):

    pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
    lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
    pro = [pro_x_list, pro_y_list, pro_z_list]
    lig = [lig_x_list, lig_y_list, lig_z_list]

    return cal_distance(pro, lig)


def cal_distance(pro, lig):

    min_d = float('inf')

    for p in zip(pro[0], pro[1], pro[2]):
        for l in zip(lig[0], lig[1], lig[2]):
            d_pl = np.sqrt(np.power(p[0]-l[0], 2)+np.power(p[1]-l[1], 2)+np.power(p[2]-l[2], 2))
            if d_pl < min_d:
                min_d = d_pl

    return min_d


def get_lig_for_protein(pro_label, lig_labels, max_distance=7, folder_name='training_data'):

    pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
    pro = [pro_x_list, pro_y_list, pro_z_list]

    lig_list = []

    for lig_label in lig_labels:
        lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
        lig = [lig_x_list, lig_y_list, lig_z_list]

        if cal_distance(pro, lig) <= max_distance:
            lig_list.append(lig_label)

    return lig_list

# bind_max_d = -1
# non_bind_min_d = 100
# labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_pro_cg.pdb"))]
#
# for i in labels:
#     for j in labels:
#         d_pl = pro_lig_distance(i,j)
#         if i == j:
#             if d_pl >bind_max_d:
#                 bind_max_d = d_pl
#                 print("pro = %s, lig = %s, bind_max_dist = %s, non_bind_min_dist = %s\n" % (i, j, bind_max_d,non_bind_min_d))
#
#         else:
#             if d_pl <non_bind_min_d:
#                 non_bind_min_d = d_pl
#                 print("pro = %s, lig = %s, bind_min_dist = %s, non_bind_min_dist = %s\n" % (i, j, bind_max_d,non_bind_min_d))

# for i in labels:
#     d_pl = pro_lig_distance(i, i)
#     if d_pl > bind_max_d:
#         bind_max_d = d_pl
#         print("pro = %s, lig = %s, bind_max_dist = %s\n" % (i, i, bind_max_d))
#
# print("bind_max_d = %s" %bind_max_d)
#print("non_bind_min_d = %s" %(non_bind_min_d))






