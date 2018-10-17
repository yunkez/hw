from read_pdb_file import *
import os
import numpy as np

pro = ['2904']
lig = ['2904','2963', '2987','2924','2944','2981', '2946', '2988','2950', '2972','2937']

for i in range(len(pro)):
    for j in range(len(lig)):
        x, y = pro_lig_reader_sample(pro_label=pro[i], lig_label=lig[j],folder_name='test_data')
        plot_3D(x)

