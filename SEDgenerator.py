"""
This file contains a Keras generator class.
The generator supplies the fit_generator function in the training script
with training and validation samples.

Each data sample consists of a spectrogram and a label array.
The spectrogram is a frequency domain representation of an audio recording, which serves as the
independent variable X; the label array  is a multidimensional array of size (Classes, Spectrogram length)
which encodes the presence of a bird species in the spectrogram frame in a binary fashion:
E.G:

                    Blackbird here
                     <------>
Blackbird  [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
Chiffchaff  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
Great tit   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
                                 <------>
                            Great Tit here

To ensure equal class representation, bird classes are sampled uniformly before picking
corresponding data samples.
"""

from glob import glob
from sklearn.model_selection import train_test_split
from os.path import join
from os.path import basename
import numpy as np
import random

from random import choice
from copy import deepcopy as copy


class SEDGenerator:

    def __init__(self, data_dir, batch_size, class_to_id, id_to_class,
                 concurrent=1, random_seed=1, train_size=0.9):
        """ Initializes the data generator.

            Args:
                data_dir (str): The directory containing numpy arrays files of data samples.
                concurrent (int): The number of data classes that will be mixed into a
                                  single sample. E.g. When training the network to recognize
                                  two species in one recording, use concurrent=2.
                class_to_id (dict): Mapping from class strings to id ints.
                id_to_class (dict): Mapping from id ints to class strings.
                batch_size (int): Number of samples in a batch.
                random_seed (int): Random seed for the train test split.
                train_size (float): The proportion of the dataset to include in the training set.
        """

        self.concurrent = concurrent
        self.data_dir = data_dir
        self.class_to_id = class_to_id
        self.id_to_class = id_to_class
        self.nr_classes = len(class_to_id)
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.train_size = train_size
        self.file_names = dict()
        self.train_data = dict()
        self.val_data = dict()
        self.species = list()

        self.data = None

        species_regex = join(data_dir, '*')

        for species_directory in glob(species_regex):
            species = basename(species_directory)
            self.species.append(species)
            sample_file_regex = join(species_directory, '*.npy')
            sample_files = sorted(glob(sample_file_regex))

            self.file_names[species] = sample_files
            self.train_data[species], self.val_data[species] = train_test_split(self.file_names[species],
                                                                                train_size=self.train_size,
                                                                                random_state=self.random_seed,
                                                                                shuffle=True)

    @property
    def steps_per_epoch(self):
        """ Defines the number of batches to process during training."""
        total_length = 0

        for species in self.train_data:
            total_length += len(self.train_data[species])

        return total_length // self.batch_size

    @property
    def steps_per_validation(self):
        """ Defines the number of batches to process during validation."""
        total_length = 0

        for species in self.val_data:
            total_length += len(self.val_data[species])

        return total_length // self.batch_size

    def generate(self, training):
        """ A generator that yields data batches."""

        if training:
            self.data = copy(self.train_data)
        else:
            self.data = copy(self.val_data)

        while True:
            batch_files = self.get_batch_files(training)
            x, y = self.batch_data(batch_files)

            yield x, y

    def get_batch_files(self, training):
        """ Samples a list of files paths for a training or validation batch.

            This is achieved popping samples off data lists. Whenever a list
            of samples has been exhausted, they are filled and shuffled.
        """
        batch_files = list()

        for sample in range(self.batch_size):
            classes_to_exclude = list()

            for concurrent_class in range(self.concurrent):
                sample_class = self.pick_with_exclusion(classes_to_exclude)
                classes_to_exclude.append(sample_class)
                bird_class = self.id_to_class[sample_class]
                samples_left = len(self.data[bird_class]) > 0

                if samples_left:
                    sample_path = self.data[bird_class].pop()

                else:
                    if training:
                        self.data[bird_class] = copy(self.train_data[bird_class])
                    else:
                        self.data[bird_class] = copy(self.val_data[bird_class])

                    random.shuffle(self.data[bird_class])
                    sample_path = self.data[bird_class].pop()

                batch_files.append(sample_path)

        return batch_files

    def batch_data(self, batch_files):
        """ Supplies a data batch given a list of files to appear in the batch."""
        x = None
        y = None
        previous_spectrogram = None
        previous_labels = None

        for layer_nr, file_path in enumerate(batch_files):
            spectrogram, labels = np.load(file_path, allow_pickle=True)
            spectrogram = np.expand_dims(spectrogram, axis=-1)
            spectrogram = np.expand_dims(spectrogram, axis=0)
            labels = np.expand_dims(labels, axis=0)

            if previous_spectrogram is not None:
                spectrogram = np.add(spectrogram, previous_spectrogram)
                labels = np.bitwise_or(labels.astype(int), previous_labels.astype(int))
                labels = labels.astype(float)

            previous_spectrogram = spectrogram
            previous_labels = labels

            if self.concurrent == 1:
                concurrent_set_complete = True
            else:
                concurrent_set_complete = not (layer_nr % (self.concurrent - 1)) and layer_nr

            if concurrent_set_complete:
                previous_spectrogram = None
                previous_labels = None

                # Stack samples into a batch
                if x is None:
                    x = spectrogram
                    y = labels
                else:
                    x = np.concatenate((x, spectrogram), axis=0)
                    y = np.concatenate((y, labels), axis=0)
        return x, y

    def pick_with_exclusion(self, to_exclude):
        """ Randomly chooses a class id while excluding already seen classes. """
        to_choose_from = [i for i in range(self.nr_classes) if i not in to_exclude]
        new_bird_class = choice(to_choose_from)
        return new_bird_class
