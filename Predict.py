from keras.models import load_model
from auc import *
from distance import *

model = load_model('AtomNet_2000x10x5x1.0.h5', custom_objects={'auc': auc})
import numpy as np

def getKey(item):
    return item[1]

dic = load_obj('pro_lig_pairs_test')

pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./testing_data_true/*_pro_cg.pdb"))]
lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./testing_data_true/*_lig_cg.pdb"))]
IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

score_dic = {}
for pro in pro_labels:

    output = {'pro': pro,
              'predicted_score': []
              }

    possible_lig_labels = get_possible_ligs_for_pro_from_dic(dic, pro)
    print('%s: %s' % (pro, possible_lig_labels))

    for lig in possible_lig_labels:
        X = np.empty((1, 48, 48, 48,4))
        X[0], y_r = pro_lig_reader_sample(pro_label=pro, lig_label=lig, folder_name='testing_data_true', test=True)
        y_p = model.predict(X)
        output['predicted_score'].append((lig, y_p[0][1]))
    output['predicted_score'] = sorted(output['predicted_score'], key=getKey, reverse=True)[:10]
    dic[pro]=output['predicted_score']
    print(output)
save_obj(dic, 'pro_lig_score')

print("done")
