from read_pdb_file import *
import os
import numpy as np


files = [f for f in os.listdir('./training_data/') if f.endswith('lig_cg.pdb')]


for f in files:
    X_list, Y_list, Z_list, atomtype_list=read_pdb("./training_data/" + f)
    lig_coords = [X_list, Y_list, Z_list]
    print(lig_coords)
    centroid = [np.mean(X_list),np.mean(Y_list),np.mean(Z_list)]
    print(centroid)
