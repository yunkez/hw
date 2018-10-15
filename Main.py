from AtomConvNet import *
from DataGenerator import *
import random
import matplotlib.pyplot as plt

grid_size = 24
num_channels = 4
grid_resolution = 0.5
grid_dim = math.floor(grid_size / grid_resolution)
batch_size = 200
num_classes = 2
validation_ratio = 0.2
t = 1
num_samples = 20
params = {'grid_size': grid_size,
          'batch_size': batch_size,
          'n_classes': num_classes,
          'grid_resolution': grid_resolution,
          'n_channels': num_channels,
          'shuffle': True}

# Datasets
# pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_pro_cg.pdb"))]
# lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_lig_cg.pdb"))]
# IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

pro_labels = [format(i + 1, '04') for i in range(100)]
lig_labels = [format(i + 1, '04') for i in range(100)]
IDs = [(pro_labels[i], lig_label) for i in range(len(pro_labels))
       for lig_label in ([lig_labels[i]]+random.sample(list(set(lig_labels)-set(pro_labels[i])), num_samples-1))]

class_weight = {
    0: 1.0,
    1: 1.0 * (num_samples-1)
}

print("starting to sample for validation partitions...")

train_steps = math.floor((1-validation_ratio) * len(IDs) / batch_size)
val_steps = math.floor(validation_ratio * len(IDs) / batch_size)
val_part = [(d[0], d[1]) for d in random.sample(IDs, math.floor(validation_ratio * len(IDs)))]
train_part = list(set(IDs) - set(val_part))

# Generators
print("starting to load data...")
training_generator = DataGenerator(train_part, **params)
validation_generator = DataGenerator(val_part, **params)

print("starting to compile model...")
model = AtomConvNet(input_shape=(grid_dim, grid_dim, grid_dim, num_channels))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on dataset
print("starting to fit model...")
history = model.fit_generator(generator=training_generator, validation_data=validation_generator,steps_per_epoch=train_steps,validation_steps=val_steps, use_multiprocessing=True,workers=10,class_weight=class_weight,verbose=1, epochs=1)

model.save('AtomNet_100x20.h5')


#print("starting to plot...")

# # Plot training & validation accuracy values
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

