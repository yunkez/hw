from keras.models import load_model
import os
import glob
from read_pdb_file import pro_lig_reader_sample
model = load_model('AtomNet.h5')
import numpy as np

pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./test_data/*_pro_cg.pdb"))]
lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./test_data/*_lig_cg.pdb"))]
IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

for pro in pro_labels:

    output = {'pro_label': pro,
              'binded_lig_label': pro,
              'predicted_lig_label': [],
              'found': False}

    for lig in lig_labels:
        X = np.empty((1, 48, 48, 48,4))
        X[0], y_r = pro_lig_reader_sample(pro_label=pro, lig_label=lig,folder_name='test_data')
        y_p = model.predict(X)
        if y_p[0][1] == 1:
            output['predicted_lig_label'].append(lig)
            if lig == pro:
                output['found'] = True
    print(output)
