"""
This file contains functions needed to generate training
data i.e. feature files, by combining audio and annotation files.
This feature data can be directly interpreted by the network.


The feature data is stored as separate files to enable the use of an arbitrarily large datasets.
The program is called as follows:

    $python3 features.py --audios_directory ~/audio_dir_here \
                         --annotations_directory' ~/annotation_dir_here \
                         --output_directory ~/output_dir_here \
                         --sequence_length 512 \
                         --sample_rate 44100 \
                         --window_length 512 \
                         --hop_length 256

How it works:

The script reads and audio file and generates a spectrogram (the independent variable).
At the same time, it reads and annotation file and generates the label array (the target variable).
All spectrograms are merged into a single extremely long spectrogram, which is then cut up
into spectrograms of length `seq_length`. The same happens for the label arrays and audio.
Finally, the spectorgrams, label array (and audio) are saved as a single tuple in a .npy file.
The .npy files can be read by the network and serve as a single training sample.
"""

import scipy.signal
import numpy as np
import librosa
from os import makedirs
from os.path import splitext
from os.path import basename
from os.path import join
from os.path import exists
from glob import glob
from tqdm import tqdm
import argparse
import uuid
import scipy.io.wavfile


def delete_audio(output_directory):
    """ Removes audio waveforms from the dataset.
        When the dataset is only used for training, and
        not for plotting mp4s videos of predictions, there
        is not need for audio to be saved in the feature files.
    """
    regex = join(output_directory, '*', '*.npy')
    feature_files = glob(regex)

    for feature_file in tqdm(feature_files):
        data = np.load(feature_file, allow_pickle=True)
        data = data[:2]
        np.save(feature_file, data)


def get_class_id_mapping(all_species):
    """ Finds the mapping between class strings such as 'blackbird'
        to class ids ints, such as 1. """
    class_id_mapping = dict()
    last_idx = 0

    for idx, species in enumerate(all_species):
        class_id_mapping[species] = idx
        last_idx = idx

    class_id_mapping['notright'] = last_idx + 1

    return class_id_mapping


def generate_annotation_dict(annotation_file):
    """ Creates a dictionary where the key is a file name
        and the value is a list containing the
            - start time
            - end time
            - bird class.
        for each annotation in that file.
    """

    annotation_dict = dict()
    for line in open(annotation_file):
        file_name, start_time, end_time, bird_class = line.strip().split('\t')

        if file_name not in annotation_dict:
            annotation_dict[file_name] = list()

        annotation_dict[file_name].append([start_time, end_time, bird_class])

    return annotation_dict


def generate_spectrogram(audio_file_path):
    """ Returns a 2D numpy array representing the spectrogram i.e.
        frequency domain representation of the audio file. """

    _, data = scipy.io.wavfile.read(audio_file_path)
    _, _, complex_coordinates = scipy.signal.stft(x=data,
                                                  fs=sample_rate,
                                                  nperseg=window_length,
                                                  noverlap=hop_length,
                                                  nfft=window_length)

    amplitudes = np.abs(complex_coordinates)
    spectrogram = librosa.amplitude_to_db(S=amplitudes,
                                          ref=np.max,
                                          amin=1e-05,
                                          top_db=85)

    # Optional normalization
    # spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.var()
    return spectrogram


def generate_label_array(label_list, spectrogram):
    """ Goes from a list of start times, end time and classes:

            [[start1, end1, class1]
             [start1, end2, class2]...]

        To a label array:
                                class1 here
                               <------>
            class1  [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            class2  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            class3  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
                                         <------>
                                        class3 Tit here
    """

    nr_frames = spectrogram.shape[1]
    nr_classes = len(class_to_id)

    label_array = np.zeros((nr_classes, nr_frames))
    label_list = np.array(label_list)

    start_times = label_list[:, 0].astype(float)
    end_times = label_list[:, 1].astype(float)
    classes = label_list[:, 2]

    start_frames = (start_times * sample_rate) / hop_length
    start_frames = np.floor(start_frames)
    start_frames = start_frames.astype(int)

    end_frames = (end_times * sample_rate) / hop_length
    end_frames = np.ceil(end_frames)
    end_frames = end_frames.astype(int)

    for idx, bird_class in enumerate(classes):
        start_frame = start_frames[idx]
        end_frame = end_frames[idx]
        class_id = class_to_id[bird_class]
        label_array[class_id, start_frame:end_frame] = 1

    return label_array


def cut_out_faults(spectrogram, label_array, wave):
    """ Cuts out frames that belong to the `not right` class and
        returns a labels array without the `not right` class. """
    not_right_id = class_to_id['notright']
    not_right_array = label_array[not_right_id]
    columns_to_be_kept = not_right_array != 1
    spectrogram = spectrogram[:, columns_to_be_kept]
    label_array = label_array[:, columns_to_be_kept]
    label_array = np.delete(label_array, not_right_id, 0)
    wavs_points_to_be_kept = [[kept] * hop_length for kept in columns_to_be_kept]
    wavs_points_to_be_kept = np.array(wavs_points_to_be_kept).flatten()
    wavs_points_to_be_kept = np.resize(wavs_points_to_be_kept, (len(wave)))
    wave = wave[wavs_points_to_be_kept]
    return spectrogram, label_array, wave


def generate_data_sample(audio_path, annotation_dict):
    """ Returns spectrogram and label array for each audio file.
        These will function as X and Y for each sample. """

    spectrogram = generate_spectrogram(audio_path)
    _, wave = scipy.io.wavfile.read(audio_path)
    base_name = basename(audio_path)
    label_list = annotation_dict[base_name]
    label_array = generate_label_array(label_list, spectrogram)
    spectrogram, label_array, wave = cut_out_faults(spectrogram, label_array, wave)
    return spectrogram, label_array, wave


def merge_data_samples(data_samples):
    """ Merges the spectrograms, labels arrays, and audio into
        long concatenated units. """

    x = None
    y = None
    z = None

    for spectrogram, label_array, wave in data_samples:

        if x is None:
            x = spectrogram
            y = label_array
            z = wave
        else:
            x = np.concatenate((x, spectrogram), 1)
            y = np.concatenate((y, label_array), 1)
            z = np.concatenate((z, wave))

    return x, y, z


def split_in_seqs(data):
    """ Splits a large spectrogram or label array into a sequence length units. """

    nr_frames = data.shape[1]
    nr_sequences = nr_frames // sequence_length
    remainder = None

    if nr_frames % sequence_length:
        last_frame_index = -(nr_frames % sequence_length)
        trimmed = data[:, :last_frame_index]
        remainder = data[:, last_frame_index:]
    else:
        trimmed = data

    sequence_wise_split = np.split(trimmed, nr_sequences, axis=1)

    return sequence_wise_split, remainder


def split_wave_in_seqs(wave):
    """ Splits a wave into length that matches the
        timespan of the spectrogram and label array. """

    nr_wave_points = wave.shape[0]
    frames_in_1_sec = sample_rate / hop_length
    nr_seconds = sequence_length / frames_in_1_sec
    wave_points_in_sequence = nr_seconds * sample_rate
    nr_sequences = nr_wave_points // wave_points_in_sequence
    remainder = None

    if nr_wave_points % wave_points_in_sequence:
        last_point = int(-(nr_wave_points % wave_points_in_sequence))
        trimmed = wave[:last_point]
        remainder = wave[last_point:]
    else:
        trimmed = wave

    sequence_wise_split = np.split(trimmed, nr_sequences)

    return sequence_wise_split, remainder


def write_samples_to_files(merged_samples, output_directory):
    """ Takes the merged samples, splits them and writes splits to files. """
    spectrograms, labels, wave = merged_samples

    sequence_spec, spec_remainder = split_in_seqs(spectrograms)
    sequence_labels, lab_remainder = split_in_seqs(labels)
    sequence_wave, wave_remainder = split_wave_in_seqs(wave)

    for sample in zip(sequence_spec, sequence_labels, sequence_wave):
        file_name = str(uuid.uuid4())
        output_path = join(output_directory, file_name)
        np.save(output_path, sample)

    remainder = (spec_remainder, lab_remainder, wave_remainder)

    return remainder


def generate_feature_files(annotation_path, audios_directory, output_directory):
    """ Generates feature files by combining extracting the labels from  annotation
        files and spectrograms from audio files. """

    annotation_dict = generate_annotation_dict(annotation_path)
    samples = list()

    # The number of samples concurrently kept in memory during
    # concatenation process. Decrease batch size if computer runs out of memory
    batch_size = 10

    for idx, file in enumerate(tqdm(annotation_dict)):

        audio_path = join(audios_directory, file)
        sample = generate_data_sample(audio_path, annotation_dict)
        samples.append(sample)

        if idx and idx % batch_size == 0:
            merged_samples = merge_data_samples(samples)
            remainder_sample = write_samples_to_files(merged_samples, output_directory)
            samples = [remainder_sample]

    merged_samples = merge_data_samples(samples)
    write_samples_to_files(merged_samples, output_directory)


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audios_directory', help='The directory containing the audio files.', required=True)
    parser.add_argument('--annotations_directory', help='The directory containg txt annotations.', required=True)
    parser.add_argument('--output_directory', help='The directory where the features are saved.', required=True)
    parser.add_argument('--sequence_length', help='The length of the sequence in frames.', type=int, required=True,
                        default=512)
    parser.add_argument('--sample_rate', help='The sample rate of the audio files.', type=int,
                        default=44100)
    parser.add_argument('--window_length', help='The length of the short-time Fourier transform widow.', type=int,
                        default=512)
    parser.add_argument('--hop_length', help='The length of the hop between each stft window.', type=int,
                        default=256)
    parser.add_argument('--include_audio', help='Whether or not to include audio wave form in feature files.'
                                                'These are typically only necessary during manual debugging and can be'
                                                'left out during training for performance.', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = setup_argparse()
    sequence_length = args.sequence_length
    sample_rate = args.sample_rate
    window_length = args.window_length
    hop_length = args.hop_length
    include_audio = args.include_audio

    regex = join(args.annotations_directory, '*.txt')
    all_annotation_files = glob(regex)
    all_species = sorted([splitext(basename(a))[0] for a in all_annotation_files])
    class_to_id = get_class_id_mapping(all_species)

    for annotation_file in tqdm(all_annotation_files):
        species, _ = splitext(basename(annotation_file))
        audio_directory = join(args.audios_directory, species)
        output_directory = join(args.output_directory, species)

        if not exists(output_directory):
            makedirs(output_directory)

        generate_feature_files(annotation_file, audio_directory, output_directory)

    if not include_audio:
        delete_audio(args.output_directory)
