import random
from read_pdb_file import *
import pickle


def cal_distance_by_label(pro_label='0001', lig_label='0001', folder_name='training_data'):
    pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
    lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
    pro = [pro_x_list, pro_y_list, pro_z_list]
    lig = [lig_x_list, lig_y_list, lig_z_list]
    return cal_distance(pro, lig)

def cal_distance(pro, lig, max_distance=7):
    min_d = float('inf')
    for p in zip(pro[0], pro[1], pro[2]):
        for l in zip(lig[0], lig[1], lig[2]):
            d_pl = np.sqrt(np.power(p[0]-l[0], 2)+np.power(p[1]-l[1], 2)+np.power(p[2]-l[2], 2))
            if d_pl < min_d:
                min_d = d_pl
            if min_d <= max_distance:
                break
    return min_d

def get_possible_lig_for_protein(pro_label, lig_labels, max_distance=7, folder_name='training_data'):
    pro_x_list, pro_y_list, pro_z_list, pro_atomtype_list = read_pdb('./' + folder_name + '/' + pro_label + '_pro_cg.pdb')
    pro = [pro_x_list, pro_y_list, pro_z_list]
    lig_list = []
    for lig_label in lig_labels:
        lig_x_list, lig_y_list, lig_z_list, lig_atomtype_list = read_pdb('./' + folder_name + '/' + lig_label + '_lig_cg.pdb')
        lig = [lig_x_list, lig_y_list, lig_z_list]
        if cal_distance(pro, lig) <= max_distance:
            lig_list.append(lig_label)
    return lig_list

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def cal_distance_and_save(start, end, filename='pro_lig_pairs'):
    dic = {}
    for i in range(start, end):
        possible_ligands = get_possible_lig_for_protein(format(i + 1, '04'), [format(j + 1, '04') for j in range(2000)])
        print('%s: %s' % (i, possible_ligands))
        dic[i] = possible_ligands
    save_obj(dic, filename)

def get_samples_for_pro(dic, pro, num_samples):
    ligs = dic[int(pro)-1]
    ligs_new = list(set(ligs) - set([pro]))
    return [pro] + random.sample(ligs_new, min(num_samples-1, len(ligs_new)))

def get_possible_ligs_for_pro_from_dic(dic, pro):
    ligs = dic[int(pro)-1]
    return ligs