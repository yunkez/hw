from AtomConvNet import *
from read_pdb_file import *
import matplotlib.pyplot as plt

grid_size = 24
num_channels = 4
grid_resolution = 0.5
grid_dim = math.floor(grid_size / grid_resolution)


model = AtomConvNet(input_shape=(grid_dim, grid_dim, grid_dim, num_channels))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_x, train_y = pro_lig_reader(grid_size, num_channels, grid_resolution)
print(">>> starting to fit model...")
history = model.fit(x=train_x, y=train_y, batch_size= 10, validation_split=0.2, epochs=10, verbose=1)


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