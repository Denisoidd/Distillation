import pathlib
import tensorflow as tf
from utils import load_config
from tensorflow.keras.models import load_model
from model import student_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

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
val_spl = config["distillation"]["train"]["val_split"]
im_h, im_w = config["distillation"]["train"]["im_h"], config["student"]["train"]["im_w"]
b_s = config["distillation"]["train"]["b_s"]
n_cl = config["distillation"]["train"]["n_classes"]
n_ep = config["distillation"]["train"]["epochs"]
temp = config["distillation"]["train"]["temperature"]
alpha = config["distillation"]["train"]["alpha"]

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
    print((train_ds))
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

# load trained teacher model
teach_model = load_model(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_teacher", compile=False)
teach_model.trainable = False
assert teach_model.trainable == False, 'Teacher model should be frozen'

# initialize student model
stud_model = student_model(im_h, im_w, n_cl)
assert stud_model.trainable == True, 'Student model should be trainable'

# create optimizer and losses
optimizer = Adam()
cross_entr = SparseCategoricalCrossentropy(from_logits=True)
kl_div = KLDivergence()

# Prepare the metrics.
train_acc_metric = SparseCategoricalAccuracy()
val_acc_metric = SparseCategoricalAccuracy()

# training distillation loop
for ep in range(n_ep):
    for step, (x, y) in enumerate(train_ds):
        # evaluate teacher model
        teacher_pred = teach_model(x)

        with tf.GradientTape() as tape:
            # forward pass of student model
            student_pred = stud_model(x, training=True)
            assert stud_model.trainable == True, 'Student model should be trainable'

            # hard labels loss
            loss_hard = cross_entr(y, student_pred)

            # soft labels loss with temperature temp
            loss_soft = kl_div(tf.nn.softmax(teacher_pred / temp), tf.nn.softmax(student_pred / temp))

            # final loss value
            loss = alpha * loss_hard + (1 - alpha) * loss_soft

        # update train accuracy metric
        train_acc_metric.update_state(y, student_pred)

        # calculate gradients
        grads = tape.gradient(loss, stud_model.weights)

        # gradient descent
        optimizer.apply_gradients(zip(grads, stud_model.trainable_weights))

        # print some info
        print("Epoch {}, step {}, loss {:5f}".format(ep, step, loss))

    # get result of train accuracy metric
    print("Train accuracy is {:4f}".format(train_acc_metric.result()))

    # reset metric
    train_acc_metric.reset_states()

    # saving model
    stud_model.save(str(pathlib.Path(__file__).parent.absolute()) + "/saved_model_distillation")

    for x_val, y_val in val_ds:
        # forward pass of student model
        student_val_pred = stud_model(x_val, training=False)
        # assert stud_model.trainable == False, 'Student model should not be trainable in val'

        # update val accuracy metric
        val_acc_metric.update_state(y_val, student_val_pred)

    # get result of train accuracy metric
    print("Val accuracy is {:4f}".format(val_acc_metric.result()))

    # reset metric
    val_acc_metric.reset_states()

