import glob
import numpy as np

def read_pdb(filename):
	
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

def pro_lig_reader():
	from keras.utils import to_categorical
	train_x = []
	train_y = []

	pro_file_names = glob.glob("*_pro_cg.pdb")


	3D_Grid = np.zeros((48,48,48)) #3D grid with 0.5A resolution and 24A x 24A x 24A size

	for i in range(len(pro_file_names)):
		label = pro_file_names[i][:4]
		pro_x = read_pdb('.training_data/' + label + '_pro_cg.pdb')
		lig_x = read_pdb('.training_data/' + label + '_lig_cg.pdb')

		centroid = np.mean(lig_x[:3],axis=0)
		pro_x[:3] = np.subtract(pro_x[:3] - centroid)
		lig_x[:3] = np.subtract(lig_x[:3] - centroid)

	train_x.append(d[0])
	train_y.append(d[0])