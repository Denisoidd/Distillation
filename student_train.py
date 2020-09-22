import pathlib
import tensorflow as tf
from utils import load_config
from model import student_model

# download dataset with flowers 5 different classes
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# check if it was well downloaded
image_count = len(list(data_dir.glob('*/*.jpg')))
if image_count:
    print("Data loaded correctly")

# load config
config = load_config(str(pathlib.Path(__file__).parent.absolute()) + "/config.yaml")
if config:
    print("Config loaded correctly")

# get values from config
val_spl = config["student"]["train"]["val_split"]
im_h, im_w = config["student"]["train"]["im_h"], config["student"]["train"]["im_w"]
b_s = config["student"]["train"]["b_s"]
n_cl = config["student"]["train"]["n_classes"]
ep = config["student"]["train"]["epochs"]

# divide data on train and validation
# load train part
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_spl,
    subset="training",
    seed=123,
    image_size=(im_h, im_w),
    batch_size=b_s)

if train_ds:
    print("Train dataset prepared successfully")

# load validation part
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=val_spl,
    subset="validation",
    seed=123,
    image_size=(im_h, im_w),
    batch_size=b_s)

if val_ds:
    print("Val dataset prepared successfully")

# in the following lines we will make sure that we use our memory properly while reading
# the images for trainig. Cache keeps images in memory after they were loaded. Prefetch
# overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# get the teacher model
stud_model = student_model(im_h, im_w, n_cl)
stud_model.summary()

# save model callback
list_of_callbacks = []
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(pathlib.Path(__file__).parent.absolute()) + config["student"]["train"]["save_path"],
    save_freq=1)
list_of_callbacks.append(model_checkpoint_callback)

# compile the model
stud_model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# train the model
history = stud_model.fit(
    train_ds,
    validation_data=val_ds,
    callbacks=list_of_callbacks,
    epochs=ep
)
