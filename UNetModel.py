from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, BatchNormalization, Dropout, UpSampling2D
from keras.models import Model
from keras import backend as K


class UNetModel:
    """
    Class for creating a standard U-Net architecture
    """
    @staticmethod
    def __down_block(x, filters, kernel_size=(3, 3), padding='same', strides=1,  kernel_initializer='glorot_uniform', batchnorm=True):
        """
        Down-sampling (Encoder) block
        :param x:
        :param filters:
        :param kernel_size:
        :param padding:
        :param strides:
        :param kernel_initializer:
        :param batchnorm:
        :return:
        """
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(x)
        if batchnorm:
            conv = BatchNormalization()(conv)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(conv)
        if batchnorm:
            conv = BatchNormalization()(conv)
        max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv)

        return conv, max_pool

    @staticmethod
    def __up_block(x, skip, filters, kernel_size=(3, 3), padding='same', strides=1,  kernel_initializer='glorot_uniform', batchnorm=True):
        """
        Up-Sampling (Decoder) block
        :param x:
        :param skip:
        :param filters:
        :param kernel_size:
        :param padding:
        :param strides:
        :param kernel_initializer:
        :param batchnorm:
        :return:
        """
        us = UpSampling2D((2, 2))(x)
        skip_con = Concatenate()([us, skip])
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(skip_con)
        if batchnorm:
            conv = BatchNormalization()(conv)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(conv)
        if batchnorm:
            conv = BatchNormalization()(conv)

        return conv

    @staticmethod
    def __bottleneck(x, filters, kernel_size=(3, 3), padding='same', strides=1, kernel_initializer='glorot_uniform', batchnorm=True):
        """
        The bottleneck part of the 'U' shape of the network
        :param x:
        :param filters:
        :param kernel_size:
        :param padding:
        :param strides:
        :param kernel_initializer:
        :param batchnorm:
        :return:
        """
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(x)
        if batchnorm:
            conv = BatchNormalization()(conv)
        conv = Conv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer, activation='relu')(conv)
        if batchnorm:
            conv = BatchNormalization()(conv)

        return conv

    def build(self, width, height, n_channels=1, filters=None, n_classes=1,  kernel_initializer='glorot_uniform', with_bn=True):
        """
        Created the U-Net network with a variable number of filters
        :param width:
        :param height:
        :param n_channels:
        :param filters:
        :param n_classes:
        :param kernel_initializer:
        :param with_bn:
        :return:
        """
        if filters is None:
            filters = [32, 64, 128, 128]

        print("[INFO] Building {}: filters = {} BN = {} ".format(self.__class__.__name__,
                                                                 filters,
                                                                 with_bn))

        inputs = Input(shape=(height, width, n_channels))

        p = inputs
        connections = []
        for f in filters[:-1]:
            c, p = self.__down_block(p, f, kernel_initializer=kernel_initializer)
            connections.append(c)

        u = self.__bottleneck(p, filters[-1], kernel_initializer=kernel_initializer)

        connections = connections[::-1]     # reverse list of connections
        filters = filters[:-1]  # eliminate bottleneck filter
        up_filters = filters[::-1]

        for c, f in zip(connections, up_filters):
            u = self.__up_block(u, c, f, kernel_initializer=kernel_initializer)

        # p0 = inputs
        # c1, p1 = self.__down_block(p0, filters[0])
        # c2, p2 = self.__down_block(p1, filters[1])
        # c3, p3 = self.__down_block(p2, filters[2])
        #
        # bn = self.__bottleneck(p3, filters[3])
        #
        # u1 = self.__up_block(bn, c3, filters[2])
        # u2 = self.__up_block(u1, c2, filters[1])
        # u3 = self.__up_block(u2, c1, filters[0])

        outputs = Conv2D(n_classes, (1, 1), padding='same', activation='sigmoid')(u)
        model = Model(inputs, outputs)

        return model

