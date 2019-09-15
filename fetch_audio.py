"""
This file contains functions for moving all annotated audio files to a separate directory.
This functionality is needed when it is not practical to share all audio files (including the
unannotated files). This is necessary as many of the audio files downloaded from xeno canto
were never annotated or skipped due to cluttered recordings.

Use as follows:
    $python3 fetch_audion.py --input_directory ~/dir_with_audio_folders_here \
                             --output_directory ~/output_dir_here \
                             --annotation_directory ~/dir_with_txt_annotations_here
"""

from os.path import join
from os.path import exists
from os.path import basename
from os.path import splitext

from os import makedirs
from shutil import copyfile
from glob import glob

import argparse


def fetch_annotated_audio_files(annotation_txt_path, input_directory, output_directory):
    """ Copies audio files occurring in an annotation file from a
        given directory to an output directory. """

    needed_files = set()

    with open(annotation_txt_path) as input_file:

        for line in input_file:
            file_name, _, _, _ = line.split('\t')
            needed_files.add(file_name)

        for file_name in needed_files:
            audio_path = join(input_directory, file_name)
            output_path = join(output_directory, file_name)

            if exists(output_path):
                continue

            if exists(audio_path):
                copyfile(audio_path, output_path)
            else:
                print('The path {} does not exist'.format(audio_path))


def fetch_all(annotation_directory, input_directory, output_directory):
    """ Calls fetch_annotated_audio_files for all annotations files in a directory. """
    regex = join(annotation_directory, '*.txt')

    for annotation_file in glob(regex):
        bird_species, _ = splitext(basename(annotation_file))
        output_bird_directory = join(output_directory, bird_species)
        input_bird_directory = join(input_directory, bird_species)

        if not exists(output_bird_directory):
            makedirs(output_bird_directory)

        fetch_annotated_audio_files(annotation_file,
                                    input_bird_directory,
                                    output_bird_directory)


if __name__ == '__main__':
    """ Takes a directory and transforms all json sound annotation files to txt annotation files. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', help='The directory containing the audio files.', required=True)
    parser.add_argument('--output_directory', help='The directory where output txt files are saved.', required=True)
    parser.add_argument('--annotation_directory', help='The directory where output txt files are saved.', required=True)
    args = parser.parse_args()
    fetch_all(args.annotation_directory, args.input_directory, args.output_directory)
