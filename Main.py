from AtomConvNet import *
from read_pdb_file import *
import matplotlib.pyplot as plt
from keras import callbacks
from DataGenerator import DataGenerator
import random

grid_size = 24
num_channels = 4
grid_resolution = 0.5
grid_dim = math.floor(grid_size / grid_resolution)
batch_size = 64
num_classes = 2
validation_ratio = 0.2

params = {'dim': (grid_dim, grid_dim, grid_dim),
          'batch_size': batch_size,
          'n_classes': num_classes,
          'n_channels': num_channels,
          'shuffle': True}

# Datasets
pro_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_pro_cg.pdb"))]
lig_labels = [os.path.basename(f)[:4] for f in sorted(glob.glob("./training_data/*_lig_cg.pdb"))]
IDs = [(pro_labels[i], lig_labels[j]) for i in range(len(pro_labels)) for j in range(len(lig_labels))]

print(">>> starting to sample for validation partitions...")

val_part = [(d[0], d[1]) for d in random.sample(IDs, math.floor(validation_ratio * len(IDs)))]
train_part = list(set(IDs) - set(val_part))
partition = {'train': train_part,
             'validation': val_part}

# Generators
print(">>> starting to load data...")
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

print(">>> starting to compile model...")
model = AtomConvNet(input_shape=(grid_dim, grid_dim, grid_dim, num_channels))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on dataset
print(">>> starting to fit model...")
history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                              use_multiprocessing=True, workers=6, verbose=1, epochs=1)

# train_x, train_y = pro_lig_reader_full(grid_size, num_channels, grid_resolution)
# history = model.fit(x=train_x, y=train_y, batch_size=100, validation_split=0.2, epochs=1, verbose=1)

print(">>> starting to plot...")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()