import pathlib
import tensorflow as tf
from utils import load_config

# download dataset with flowers 5 different classes
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# check if it was well downloaded
image_count = len(list(data_dir.glob('*/*.jpg')))
if image_count:
    print("Data loaded correctly")

# load config
config = load_config("config.yaml")
if config:
    print("Config loaded correctly")

# divide data on train and validation
# load train part
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=config["teacher"]["train"]["val_split"],
  subset="training",
  seed=123,
  image_size=(config["teacher"]["train"]["im_h"], config["teacher"]["train"]["im_w"]),
  batch_size=config["teacher"]["train"]["b_s"])

if train_ds:
    print("Train dataset prepared successfully")

# load validation part
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=config["teacher"]["train"]["val_split"],
  subset="validation",
  seed=123,
  image_size=(config["teacher"]["train"]["im_h"], config["teacher"]["train"]["im_w"]),
  batch_size=config["teacher"]["train"]["b_s"])

if val_ds:
    print("Val dataset prepared successfully")




