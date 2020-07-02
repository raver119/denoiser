"""
This file contains training setup for colorizer neural network
"""
import os
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.layers as l
from typing import Dict, Any
from world import FramesGenerator, DataSetsGenerator, PatchedDataSetsGenerator, TensorFlowWrapper

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("TensorFlow version: %s" % tf.__version__)

batch_size = 256
num_epochs = 1
grid_search = False
parallel = False

patch_height = 64
patch_width = 64
patch_stride = 16

tf.compat.v1.disable_eager_execution()


def build_model(params: Dict[str, Any]) -> tf.keras.Model:
    """
    This method returns a Model
    :return:
    """
    dirty = k.Input(shape=(3, patch_height, patch_width), dtype='float32', name='dirty')
    x_root = l.Conv2D(filters=32, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(dirty)
    x = l.Conv2D(filters=16, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(x_root)
    x = l.Conv2DTranspose(filters=32, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(x)
    x = l.Concatenate(1, name="x_concat")([x_root, x])
    x = l.Conv2DTranspose(filters=64, kernel_size=3, padding='same', data_format='channels_first', activation=params['last_activation'])(x)

    y_root = l.Conv2D(filters=32, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(dirty)
    y_16 = l.Conv2D(filters=16, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(y_root)
    y = l.Conv2D(filters=8, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(y_16)
    y = l.Conv2DTranspose(filters=16, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(y)
    y = l.Concatenate(1, name="y16_concat")([y_16, y])
    y = l.Conv2DTranspose(filters=32, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['conv_activation'])(y)
    y = l.Concatenate(1, name="y_concat")([y_root, y])
    y = l.Conv2DTranspose(filters=64, kernel_size=3, padding='same', data_format='channels_first', activation=params['last_activation'])(y)

    z = l.Concatenate(1, name="z_concat")([x, y])
    clean = l.Conv2DTranspose(filters=3, kernel_size=3, strides=[2, 2], padding='same', data_format='channels_first', activation=params['out_activation'], name='clean')(z)

    m = k.Model(inputs=[dirty], outputs=[clean])
    optimizer = k.optimizers.RMSprop(learning_rate=params['lr'])
    m.compile(optimizer=optimizer, loss={'clean': params['loss']},
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return m


# setup generator
datasets = PatchedDataSetsGenerator("/data/px/frames", batch_size=batch_size,
                                    height=patch_height, width=patch_width, stride=patch_stride,
                                    test_images=2, train_images=200)

# setting up endless generators for TF
train_wrapper = TensorFlowWrapper(datasets.raw_train)
test_wrapper = TensorFlowWrapper(datasets.raw_test)

# setup TF Datasets
train_dataset = tf.data.Dataset.from_generator(train_wrapper.callable, output_types=(tf.float32, tf.float32),
                                               output_shapes=(tf.TensorShape((None, 3, patch_height, patch_width)),
                                                              tf.TensorShape((None, 3, patch_height, patch_width))))

test_dataset = tf.data.Dataset.from_generator(test_wrapper.callable, output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape((None, 3, patch_height, patch_width)),
                                                             tf.TensorShape((None, 3, patch_height, patch_width))))

print("Train batches: %i; Test batches: %i" % (datasets.train_size(), datasets.test_size()))
if parallel:
    pass
else:
    # dict is used for grid search compatibility & better usability
    model = build_model({'lr': 0.001,
                         'conv_activation': 'swish',
                         'last_activation': 'tanh',
                         'out_activation': 'sigmoid',
                         'loss': 'mean_squared_error'})
    model.summary()

    # train & save model
    model.fit(train_dataset, verbose=1, steps_per_epoch=datasets.train_size(), epochs=num_epochs,
              validation_data=test_dataset, validation_steps=datasets.test_size())
    model.save('/data/models/denoiser.h5')

