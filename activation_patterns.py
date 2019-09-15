"""
This file contains code for saving the output of intermediate layers.
It can be used to see what kind of feature vectors are produced throughout
the network for a sample.

This enbles the search for interesting network patterns.
For example: In this research the t-sne of the feature vectors
for single time frames were similar if they belonged to the same song phrases.
These vectors were measured in the recurrent layers.
"""

import numpy as np
from keras import backend
from sklearn.manifold import TSNE


def draw_activation_pattern(ax, model, spectrogram, feature_layer_name='reshape_1'):
    """ Draws a t-sne of a layer's feature vector to a matplotlib plot.

        Args:
            ax: A matplotlib axis to plot to.
            model: The neural network used to make predictions.
            spectrogram: A spectrogram of an audio fragment.
            feature_layer_name: The name of the layer whose output vector will be visualized.
    """

    feature_layer = model.get_layer(feature_layer_name).output
    input_placeholder = model.input
    trimmed_network = backend.function([input_placeholder], [feature_layer])
    spectrogram = [spectrogram]
    output_batch = trimmed_network(spectrogram)[0]
    for output in output_batch:
        squeezed = output
        embedding = TSNE(n_components=1).fit_transform(squeezed)
        ys = embedding.flatten()
        ys = (ys - ys.min()) / (ys.max() - ys.min())
        nr_frames = ys.shape[0]
        colors = ys
        xs = np.arange(0, nr_frames)
        ax.scatter(xs, ys, c=colors)
