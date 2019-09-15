"""
This file contains code for initiating a training session for
a convolutional recurrent neural network. First a model is built
according to a user chosen layer specification. Thereafter, a generator
is initialized that will provide data batches for both training and
validation steps.  Training progress is reported to a tensorboard log
and model checkpoints are made throughout training for the best
performing models (according to val_loss).
"""

from alt_model_checkpoint import AltModelCheckpoint
from keras.utils import multi_gpu_model
from keras.layers import Bidirectional, TimeDistributed, Conv2D, MaxPooling2D,\
                         Input, GRU, Dense, Activation, Dropout, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from time import time
from keras.callbacks import TensorBoard
from SEDgenerator import SEDGenerator

from os.path import join
from glob import glob
from os.path import basename
import keras.backend as K

from metrics import f1_overall_framewise, er_overall_framewise


K.set_learning_phase(1)


def build_model(sample, cnn_filters, max_pool_sizes, rnn_sizes, fc_sizes, dropout_rate, multi_gpu):
    """ Builds a convolutional recurrent neural network according to some specification."""
    data_in, data_out = sample
    sequence_length = data_in.shape[-2]
    nr_classes = data_out.shape[-2]

    model_sizes = zip(cnn_filters, max_pool_sizes)
    spec_start = Input(shape=(data_in.shape[1], data_in.shape[2], 1))
    spec_x = spec_start

    for filters_size, pool_size in model_sizes:
        spec_x = Conv2D(filters=filters_size, kernel_size=(3, 3), padding='same')(spec_x)
        spec_x = BatchNormalization()(spec_x)
        spec_x = Activation('relu')(spec_x)
        spec_x = MaxPooling2D(pool_size=(pool_size, 1))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((sequence_length, -1))(spec_x)
    for rnn_size in rnn_sizes:
        spec_x = Bidirectional(GRU(rnn_size,
                                   activation='tanh',
                                   dropout=dropout_rate,
                                   recurrent_dropout=dropout_rate,
                                   return_sequences=True), merge_mode='mul')(spec_x)

    for fc_size in fc_sizes:
        spec_x = TimeDistributed(Dense(fc_size))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(nr_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)
    out = Permute((2, 1))(out)
    base_model = Model(inputs=spec_start, outputs=out)

    if multi_gpu:
        model_to_train = multi_gpu_model(base_model, gpus=multi_gpu)
    else:
        model_to_train = base_model

    metrics = ['mae', 'mse', 'accuracy', f1_overall_framewise, er_overall_framewise]
    model_to_train.compile(optimizer='Adam', loss='binary_crossentropy', metrics=metrics)
    model_to_train.summary()

    return base_model, model_to_train


def get_dicts(data_directory):
    """ Creates a two dictionaries, one mapping class id's to names,
        the other mapping class names to id's.
    """
    regex = join(data_directory, '*')
    directories = glob(regex)
    all_species = [basename(directory) for directory in directories]
    all_species = sorted(all_species)
    class_to_id = dict()
    id_to_class = dict()

    for idx, species in enumerate(all_species):
        class_to_id[species] = idx
        id_to_class[idx] = species

    return class_to_id, id_to_class


def main():
    """ Trains the network, saves model checkpoints and logs training progress."""
    multi_gpu = 6
    batch_size = 30
    epochs = 100
    cnn_filters = [64, 64, 128, 128, 256]
    max_pool_sizes = [2, 2, 2, 2, 2]
    rnn_sizes = [256, 256]
    fc_sizes = [256]
    dropout_rate = 0.0

    data_directory = 'features'
    class_to_id, id_to_class = get_dicts(data_directory)

    generator = SEDGenerator(data_dir=data_directory,
                             concurrent=2,
                             class_to_id=class_to_id,
                             id_to_class=id_to_class,
                             batch_size=batch_size,
                             random_seed=1,
                             train_size=0.9)

    training_generator = generator.generate(training=True)
    testing_generator = generator.generate(training=False)
    sample = next(training_generator)

    base_model, model_to_train = build_model(sample,
                                             cnn_filters=cnn_filters,
                                             max_pool_sizes=max_pool_sizes,
                                             rnn_sizes=rnn_sizes,
                                             fc_sizes=fc_sizes,
                                             dropout_rate=dropout_rate,
                                             multi_gpu=multi_gpu)

    log_dir = 'tensorboard_logs/{}'.format(time())
    tensorboard = TensorBoard(log_dir=log_dir)
    checkpointer = AltModelCheckpoint('model_checkpoint.h5', base_model, verbose=1, save_best_only=True)
    callbacks_list = [tensorboard, checkpointer]

    model_to_train.fit_generator(generator=training_generator,
                                 validation_data=testing_generator,
                                 steps_per_epoch=generator.steps_per_epoch,
                                 validation_steps=generator.steps_per_validation,
                                 callbacks=callbacks_list,
                                 epochs=epochs)

    base_model.save('model_final_epoch.h5')


if __name__ == '__main__':
    main()
