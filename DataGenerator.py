import keras
from read_pdb_file import *

class DataGenerator(keras.utils.Sequence):
    #Generates data for Keras
    def __init__(self, list_IDs, batch_size=48, grid_size=24, grid_resolution=0.5, n_channels=4,
                 n_classes=2, shuffle=True):
        # Initialization
        self.grid_size = grid_size
        self.grid_dim = math.floor(grid_size/grid_resolution)
        self.grid_resolution = grid_resolution
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        #Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        Generate indexes of the batch
        """
        #print("%d / %d" %(index, np.floor(len(self.list_IDs) / self.batch_size)))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        """
        Generates data containing batch_size samples'
        X : (n_samples, *dim, n_channels)
        """

        # Initialization
        X = np.empty((self.batch_size, self.grid_dim, self.grid_dim, self.grid_dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i], y[i] = pro_lig_reader_sample(ID[0], ID[1], self.grid_size, self.n_channels, self.grid_resolution)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

