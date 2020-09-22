import pathlib

from tensorflow.keras.models import Sequential, load_model
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
        data_augmentation_layer(im_h, im_w),
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(im_h, im_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_cl)
    ])


def student_model(im_h, im_w, n_cl):
    """
    Reduced from teacher_model
    :param im_h: image height
    :param im_w: image width
    :param n_cl: number of flowers classes
    :return: sequential model
    """
    return Sequential([
        data_augmentation_layer(im_h, im_w),
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(im_h, im_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_cl)
    ])


def data_augmentation_layer(im_h, im_w):
    """
    Data augmentation layer to fight with overfitting
    :param im_h: image height
    :param im_w: image width
    :return: sequential model for data augmentation
    """
    return Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(im_h, im_w, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )


# def distillation_model(im_h, im_w, n_cl):
#     # load trained teacher model and freeze it
#     teach_model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_teacher", compile=False)
#     teach_model.trainable = False
#
#     # check freeze
#     assert teacher_model.trainable == False, 'teacher model should be frozen'
#
#     # create student network
#     stud_model = student_model(im_h, im_w, n_cl)


