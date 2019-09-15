"""
This file contains code for a creating visualizations of network predictions.
This code can be used both for creating static images to inspect
a network's performance, and as a keras callback to view
how the network's predictions change over time during training.
"""

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from activation_patterns import draw_activation_pattern
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import io
import uuid
from SEDgenerator import SEDGenerator
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import moviepy.editor as mpe
import matplotlib.animation as animation
import scipy
import scipy.io.wavfile
from os.path import join
from os.path import basename
from glob import glob


def get_spans(labels):
    """ Gives a set of spans given a label array. """
    labels[labels >= 0.5] = 1
    all_spans = list()
    for bird_class in labels:
        spans = list()
        row = bird_class
        start = 0
        currently_in_annotation = False

        for idx, element in enumerate(row):
            if int(element) == 1:
                if not currently_in_annotation:
                    currently_in_annotation = True
                    start = idx
                else:
                    continue

            elif int(element) == 0:
                if currently_in_annotation:
                    end = idx
                    length = end - start
                    spans.append((start, length))
                    currently_in_annotation = False
                else:
                    continue

        if int(element) == 1 and currently_in_annotation:
            end = idx
            length = end - start
            spans.append((start, length))

        all_spans.append(spans)
    return all_spans


def draw_labels(ax, frame_spans, colors, name_list):
    """ Plots horizontal bars in a given frame to create a Grannt plot."""

    for idx, bird_frame_spans in enumerate(frame_spans):
        start_height = 0.5*len(frame_spans) - (idx + 1)* 0.5
        thickness = 0.5
        x_ranges = bird_frame_spans
        y_ranges = (start_height, thickness)
        ax.broken_barh(x_ranges, y_ranges, facecolors=colors[idx], label=name_list[idx])


def draw_confidence(ax, labels, colors):
    for idx, row in enumerate(labels):
        ax.plot(row, c=colors[idx])


def draw_spectrogram(ax, spectrogram):
    spectrogram = np.squeeze(spectrogram)
    ax.pcolormesh(spectrogram)


def unpack(data):
    data = data[0]
    data = np.squeeze(data)
    return data


def update(current_draw, line1, nr_frames, number_of_updates):
    """ Draws the vertical line that moves through
        the plots from left to right. """

    proportion_finished = current_draw / number_of_updates
    current_frame = int(nr_frames * proportion_finished)
    line_coord = [current_frame, current_frame]
    line1.set_data(line_coord, [0, 257])

    return line1


def plot_sample(fig, sample, model, name_list, video):
    """ Plots a sample and the models predictionas either
        a static image or an animation mp4 with audio.
    """
    colors = ['purple', 'blue', 'red', 'green', 'orange']
    spectrogram, labels, wave = sample
    labels = np.squeeze(labels)
    prediction = unpack(model.predict(spectrogram, steps=1))
    ground_truth_spans = get_spans(labels)
    prediction_spans = get_spans(prediction)
    spectogram_plot = plt.subplot2grid((5, 1), (0, 0))
    ground_truth_plot = plt.subplot2grid((5, 1), (1, 0))
    prediction_plot = plt.subplot2grid((5, 1), (2, 0))
    confidence_plot = plt.subplot2grid((5, 1), (3, 0))
    tsne_plot = plt.subplot2grid((5, 1), (4, 0))

    draw_spectrogram(ax=spectogram_plot, spectrogram=spectrogram[0])
    draw_labels(ax=ground_truth_plot, frame_spans=ground_truth_spans, name_list=name_list, colors=colors)
    draw_labels(ax=prediction_plot, frame_spans=prediction_spans, name_list=name_list, colors=colors)
    draw_confidence(ax=confidence_plot, labels=prediction, colors=colors)
    draw_activation_pattern(ax=tsne_plot, model=model, spectrogram=spectrogram)

    spectogram_plot.set_title('Spectrogram')
    ground_truth_plot.set_title('Ground truth')
    prediction_plot.set_title('Prediction')
    confidence_plot.set_title('Confidence')
    tsne_plot.set_title('Feature T-SNE')

    spectogram_plot.axes.get_yaxis().set_visible(False)
    ground_truth_plot.axes.get_yaxis().set_visible(False)
    prediction_plot.axes.get_yaxis().set_visible(False)

    ground_truth_plot.set_xlim(left=0, right=labels.shape[1])
    ground_truth_plot.set_ylim(bottom=0, top=0.5*labels.shape[0])
    prediction_plot.set_xlim(left=0, right=labels.shape[1])
    prediction_plot.set_ylim(bottom=0, top=0.5*labels.shape[0])
    confidence_plot.set_ylim(bottom=0, top=1.3)
    ground_truth_plot.legend(loc=(1.04, 0.2))

    if video:
        line1 = mlines.Line2D([0, 0], [0, 257])
        spectogram_plot.add_line(line1)

        nr_seconds = len(wave) / 44100
        fps = 30
        seconds_to_milliseconds = 1000
        interval = (1 / fps) * seconds_to_milliseconds
        nr_frames = labels.shape[1]
        number_of_updates = int(fps * nr_seconds)

        ani = FuncAnimation(fig,
                            update,
                            frames=number_of_updates,
                            interval=interval,
                            fargs=(line1, nr_frames, number_of_updates))

        writer = animation.FFMpegFileWriter(fps=30)
        ani.save('lines.mp4', writer=writer)
        scipy.io.wavfile.write('temp_audio.wav', 44100, wave)
        audio = mpe.AudioFileClip("temp_audio.wav")
        video1 = mpe.VideoFileClip("lines.mp4")
        final = video1.set_audio(audio)
        final.write_videofile("demo{}.mp4".format(uuid.uuid4()))


def draw_sample_visualization(sample, model, name_list):
    """ Draws a spectrogram, ground truth, prediction, t-sne plot for a sample."""
    fig = plt.figure(figsize=(7, 7))
    plot_sample(fig, sample, model, name_list=name_list, video=False)
    fig.tight_layout()
    img = fig2img(fig)
    plt.close()
    width, height = img.size
    channel = 3
    output = io.BytesIO()
    img.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    summary = tf.Summary.Image(height=height,
                               width=width,
                               colorspace=channel,
                               encoded_image_string=image_string)
    return summary


def draw_sample_visualization_non_callback(sample, model, name_list):
    """ Draws a spectrogram, ground truth, prediction, t-sne plot for a sample."""
    fig = plt.figure(figsize=(7, 7))
    plot_sample(fig, sample, model, name_list=name_list, video=False)
    fig.tight_layout()
    plt.savefig('{}'.format(uuid.uuid4()))


def fig2img(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    im = Image.frombytes("RGBA", (width, height), s)
    return im


class PlotCallback(keras.callbacks.Callback):

    def __init__(self, log_dir, base_model, generator, number_of_plots):
        super().__init__()
        self.log_dir = log_dir
        self.base_model = base_model
        self.generator = generator.generate(training=False)
        self.sample_list = next(self.generator)
        self.name_list = generator.species
        self.number_of_plots = number_of_plots

    def on_epoch_end(self, epoch, logs={}):
        writer = tf.summary.FileWriter(self.log_dir)
        spectrograms = self.sample_list[0]
        spectrograms = spectrograms[:self.number_of_plots]
        labels = self.sample_list[1]
        labels = labels[:self.number_of_plots]
        for idx, sample in enumerate(zip(spectrograms, labels)):
            image_summary = draw_sample_visualization(sample, self.model, self.name_list)
            summary = tf.Summary(value=[tf.Summary.Value(tag=str(idx), image=image_summary)])
            writer.add_summary(summary, epoch)


        writer.close()


def get_dicts(data_directory):
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


if __name__ == '__main__':
    data_directory = 'features'
    class_to_id, id_to_class = get_dicts(data_directory)

    generator = SEDGenerator(data_dir=data_directory,
                             concurrent=3,
                             class_to_id=class_to_id,
                             id_to_class=id_to_class,
                             batch_size=1,
                             random_seed=2,
                             train_size=0.9)

    testing_generator = generator.generate(training=False)
    name_list = sorted(generator.species)
    model = keras.models.load_model('model.h5')

    for i in range(5):
        sample = next(testing_generator)
        draw_sample_visualization_non_callback(sample, model, name_list)
