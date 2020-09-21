import pathlib
import tensorflow as tf

# download dataset with flowers 5 different classes
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# check if it was well downloaded
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
