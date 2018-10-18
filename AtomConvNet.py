from keras.models import Model
from keras.layers import Input, Dense, Conv3D, Activation, MaxPooling3D, Dropout, Flatten


def AtomConvNet(input_shape=(48, 48, 48, 4)):
    im_input = Input(shape=input_shape)

    t = Conv3D(24, (3, 3, 3), activation='relu', padding='same')(im_input)
    t = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(t)

    t = Conv3D(48, (3, 3, 3), activation='relu', padding='same')(t)
    t = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(t)

    t = Conv3D(96, (3, 3, 3), activation='relu', padding='same')(t)
    t = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(t)

    t = Conv3D(192, (3, 3, 3), activation='relu', padding='same')(t)
    t = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(t)

    t = Flatten()(t)

    t = Dense(192, activation='relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(96, activation='relu')(t)
    t = Dropout(0.5)(t)

    t = Dense(2)(t)
    output = Activation('softmax')(t)

    model = Model(inputs=im_input, outputs=output)

    return model
