from AtomConvNet import *
from DataGenerator import *
import random
import matplotlib.pyplot as plt
from AUC import *

grid_size = 24
num_channels = 4
grid_resolution = 0.5
grid_dim = math.floor(grid_size / grid_resolution)
batch_size = 100
num_classes = 2
validation_ratio = 0.1
num_samples = 10
num_epochs = 1
params = {'grid_size': grid_size,
          'batch_size': batch_size,
          'n_classes': num_classes,
          'grid_resolution': grid_resolution,
          'n_channels': num_channels,
          'shuffle': True}

# Datasets
# labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_pro_cg.pdb"))]
labels = [format(i + 1, '04') for i in range(2000)]

val_labels = random.sample(labels, math.floor(validation_ratio * len(labels)))
train_labels = list(set(labels) - set(val_labels))


class_weight = {
    0: 1.0,
    1: 1.0 * (num_samples-1)
}

print("starting to sample for validation partitions...")

train_part = [(train_labels[i], lig_label) for i in range(len(train_labels))
              for lig_label in ([train_labels[i]]+random.sample(list(set(train_labels)-set(train_labels[i])), num_samples-1))]

val_part = [(val_labels[i], lig_label) for i in range(len(val_labels))
            for lig_label in ([val_labels[i]]+random.sample(list(set(val_labels)-set(val_labels[i])), num_samples-1))]

train_steps = math.floor(1.0 * len(train_part) / batch_size)
val_steps = math.floor(1.0 * len(val_part) / batch_size)

# Generators
print("starting to load data...")
training_generator = DataGenerator(train_part, **params)
validation_generator = DataGenerator(val_part, **params)

print("starting to compile model...")
model = AtomConvNet(input_shape=(grid_dim, grid_dim, grid_dim, num_channels))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[auc])

# Train model on dataset
print("starting to fit model...")
history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                              steps_per_epoch=train_steps, validation_steps=val_steps,
                              use_multiprocessing=True, class_weight=class_weight,
                              workers=4, verbose=1, epochs=num_epochs)

model.save('AtomNet_%sx%sx%s.h5' %(len(labels),num_samples,num_epochs))

print("starting to plot...")

# # Plot training & validation accuracy values
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('Model AUC')
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

