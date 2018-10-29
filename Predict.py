from keras.models import load_model
from AUC import *
from DistanceUtil import *
import numpy as np

def getKey(item):
    return item[1]

model = load_model('AtomNet_2000x10x5x1.0.h5', custom_objects={'auc': auc})

dic = load_obj('pro_lig_pairs_test')

#pro_labels = ['0030', '0035', '0038', '0052', '0057', '0063', '0067', '0090', '0105', '0111', '0117', '0118', '0124', '0157', '0162', '0184', '0194', '0246', '0267', '0295', '0306', '0307', '0310', '0314', '0344', '0358', '0406', '0412', '0438', '0444', '0449', '0467', '0487', '0504', '0520', '0564', '0570', '0574', '0599', '0601', '0605', '0616', '0638', '0639', '0654', '0666', '0684', '0706', '0722', '0725', '0753', '0754', '0812', '0815', '0821', '0824']
pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./testing_data_release/*_pro_cg.pdb"))]
lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./testing_data_release/*_lig_cg.pdb"))]
IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

score_dic = {}
for pro in pro_labels:

    output = {'pro': pro,
              'predicted_score': []
              }

    possible_lig_labels = get_possible_ligs_for_pro_from_dic(dic, pro)
    #possible_lig_labels = lig_labels
    print('%s: %s' % (pro, possible_lig_labels))

    for lig in possible_lig_labels:
        X = np.empty((1, 48, 48, 48,4))
        X[0], y_r = pro_lig_reader_sample(pro_label=pro, lig_label=lig, folder_name='testing_data_release', test=True)
        y_p = model.predict(X)
        output['predicted_score'].append((lig, y_p[0][1]))
    output['predicted_score'] = sorted(output['predicted_score'], key=getKey, reverse=True)[:10]
    score_dic[pro]=output['predicted_score']
    print(output)
save_obj(score_dic, 'pro_lig_score')

print("done")

# dic = load_obj('pro_lig_score')

with open('test_predictions.txt', 'w') as f:
    header = '\t'.join(['pro_id','lig1_id','lig2_id','lig3_id','lig4_id','lig5_id','lig6_id','lig7_id','lig8_id','lig9_id','lig10_id'])
    f.write(header + '\n')
    for pro in [format(i + 1, '04') for i in range(len(pro_labels))]:
        row = [int(pro)] + [int(i[0]) for i in score_dic[pro]]
        s = '\t'.join([str(x) for x in row])
        f.write(s + '\n')