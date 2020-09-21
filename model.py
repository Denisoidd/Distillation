from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def teacher_model(im_h, im_w, n_cl):
    """
    It's a simple sequential model for flowers classification problem.
    It was mostly inspired by https://www.tensorflow.org/tutorials/images/classification
    :param im_h: image height
    :param im_w: image width
    :param n_cl: number of flowers classes
    :return: sequential model
    """
    return Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(im_h, im_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_cl)
    ])
