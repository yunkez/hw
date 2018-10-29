from keras.models import load_model
from AUC import *
from DistanceUtil import *
import numpy as np

def getKey(item):
    return item[1]

grid_size = 24
grid_resolution = 0.5
grid_dim = math.floor(grid_size / grid_resolution)

model = load_model('AtomNet_3000x10x10x1.0x24x0.5.h5', custom_objects={'auc': auc})

dic = load_obj('pro_lig_pairs_eval')

pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./test_data/*_pro_cg.pdb"))]
lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./test_data/*_lig_cg.pdb"))]
IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

correct = 0
for pro in pro_labels:

    output = {'pro': pro,
              'found': False,
              'predicted_score': []
              }

    possible_lig_labels = get_possible_ligs_for_pro_from_dic(dic, pro)
    print('%s: %s' % (pro, possible_lig_labels))

    for lig in possible_lig_labels:
        X = np.empty((1, grid_dim, grid_dim, grid_dim,4))
        X[0], y_r = pro_lig_reader_sample(pro_label=pro, lig_label=lig, folder_name='test_data')
        y_p = model.predict(X)
        output['predicted_score'].append((lig, y_p[0][1]))
    output['predicted_score'] = sorted(output['predicted_score'], key=getKey, reverse=True)[:10]
    output['found'] = True if pro in [a[0] for a in output['predicted_score']] else False
    if output['found']: correct += 1
    print(output)

print("accuracy: %s" % (correct / len(pro_labels) * 1.0))
print("done")

