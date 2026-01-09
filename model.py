import os, tensorflow as tf, numpy as np, matplotlib.pyplot as plt, shutil, cv2, random
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset")
od.download("https://www.kaggle.com/datasets/prasunroy/natural-images")


class Config():
  SRC_FACE = ["/content/human-faces-dataset/Human Faces Dataset/AI-Generated Images"]
  SRC_NO_FACE = ["/content/natural-images/natural_images"]
  DST_DIR = "dst"
  DST_FACE = "dst/face"
  DST_NO_FACE = "dst/noface"
  EXCLUDE = ("person", "motorbike")
  EXT = (".jpg", ".jpeg")

  IMAGE_SIZE = (256, 256)
  BATCH_SIZE = 32
  VALIDATION_SPLIT = 0.2
  HEAD_EPOCHS = 10
  EPOCHS = 5
  HEAD_LR = 1e-4
  LR = 1e-5
  MIN_LR = 1e-7
  FINE_TUNE_AT = 100
  PATIENCE = 4
  LR_REDUCE_PATIENCE = 3
  LR_REDUCE_FACTOR = 0.2
  SEED = 69
  STDDEV = 0.3
  CLASS_NAMES = ["face", "noface"]
  TRAIN_SIZE = 0.7
  VAL_SIZE = 0.2
  THRESHOLD = {'val_auc': 0.99, 'val_precision': 0.98}
  ESG_PATIENCE = 3
  AUTOTUNE = tf.data.AUTOTUNE

def copy_data(src, dst):
  for path in (Config.DST_FACE, Config.DST_NO_FACE):
    os.makedirs(path, exist_ok=True)

  for path in src:
    copied = 0
    if not os.path.exists(path):
      continue
    for root, dirs, files in os.walk(path):
      if os.path.basename(root) in Config.EXCLUDE:
        continue
      for name in files:
        if not name.lower().endswith(Config.EXT):
          continue
        shutil.copy2(os.path.join(root, name), os.path.join(dst, name))
        copied += 1
  print(copied)


def normalization(x, y, face_index):
  # x = tf.cast(x, tf.float32) / 255.0
  # y = tf.cast(tf.equal(y, face_index), tf.float32)
  return x, y

class CreateNoise(tf.keras.Layer):
  def __init__(self) -> None:
    super().__init__()

  def call(self, images, training=None):
    if not training:
      return

    noise = tf.random.normal(shape = tf.shape(images), mean=0.0, stddev=Config.STDDEV*255)
    noisy_images = tf.cond(tf.random.uniform([]) < 0.6,  
    )
    return tf.clip_by_value(noisy_images, 0, 255)



def get_augmentation():
    return tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1),
  tf.keras.layers.RandomZoom(0.4),
  CreateNoise(),
  ])
augmentation = get_augmentation()


def create_dataset():
  dataset = tf.keras.utils.image_dataset_from_directory(Config.DST_DIR, image_size=Config.IMAGE_SIZE, batch_size=None, class_names=Config.CLASS_NAMES, seed=Config.SEED, shuffle=True, label_mode='binary')
  dataset_size = tf.data.experimental.cardinality(dataset).numpy()
  train_size = int(Config.TRAIN_SIZE * dataset_size)
  val_size = int(Config.VAL_SIZE * dataset_size)

  train_ds = dataset.take(train_size)
  val_test_together = dataset.skip(train_size)
  val_ds = val_test_together.take(val_size)
  test_ds = val_test_together.skip(val_size)

  face_index = Config.CLASS_NAMES.index("face")
  train_ds = train_ds.map(lambda x, y: normalization(x, y, face_index))
  val_ds = val_ds.map(lambda x, y: normalization(x, y, face_index))
  test_ds = test_ds.map(lambda x, y: normalization(x, y, face_index))

  train_ds = train_ds.batch(Config.BATCH_SIZE)
  val_ds = val_ds.batch(Config.BATCH_SIZE)
  test_ds = test_ds.batch(Config.BATCH_SIZE)

  return train_ds, val_ds, test_ds

def show_dataset(train_ds):
  for image, label in train_ds.take(1):
    for i in range(Config.BATCH_SIZE):
      plt.imshow(image[i].numpy().astype("uint8"))
      plt.show()



def build_model():
  base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*Config.IMAGE_SIZE, 3))
  base_model.trainable = False

  inputs = tf.keras.Input(shape=(*Config.IMAGE_SIZE, 3))
  x = augmentation(inputs)
  x = base_model(x, training=False)

  x = tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
  x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
  x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)
  x = tf.keras.layers.Dropout(0.2, name='dropout_2')(x)
  outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

  model = tf.keras.Model(inputs, outputs, name='face_detector')
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.HEAD_LR), loss = "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.AUC(name='auc')])

  return model, base_model


class EarlyStoppingGood(tf.keras.callbacks.Callback):
  def __init__(self) -> None:
    super().__init__()
    self.wait = 0

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    all_metrics_good = True

    for metric, threshold in Config.THRESHOLD.items():
      current_value = logs.get(metric)
      if current_value is None or current_value < threshold:
        all_metrics_good = False
        break
    if all_metrics_good:
      self.wait += 1
      if self.wait >= Config.ESG_PATIENCE:
        self.model.stop_training = True
    else:
      self.wait = 0


def get_callbacks():
  callbacks = [
      EarlyStopping(monitor="val_loss", patience=Config.PATIENCE, restore_best_weights=True),
      ReduceLROnPlateau(monitor="val_loss", factor=Config.LR_REDUCE_FACTOR, patience=Config.LR_REDUCE_PATIENCE, min_lr=Config.MIN_LR),
      EarlyStoppingGood()
      ]
  return callbacks

def train_head(model, train_ds, val_ds):
  history = model.fit(train_ds, validation_data = val_ds, epochs = Config.HEAD_EPOCHS, callbacks=get_callbacks())
  return history


def train_all(model, base_model, train_ds, val_ds, test_ds):
  base_model.trainable = True
  for layer in base_model.layers[:Config.FINE_TUNE_AT]:
    layer.trainable = False

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LR), loss = "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.AUC(name='auc')])
  history = model.fit(train_ds, validation_data = val_ds, epochs = Config.EPOCHS, callbacks=get_callbacks())
  model.evaluate(test_ds)
  return history



def predict_image(path, model):
  img = cv2.imread(path)
  img = cv2.resize(img, Config.IMAGE_SIZE)
  img = np.expand_dims(img, axis=0)
  prediction = model.predict(img)[0][0]
  print(prediction)


def main():
  np.random.seed(Config.SEED)
  tf.random.set_seed(Config.SEED)

  train_ds, val_ds, test_ds = create_dataset()
  show_dataset(train_ds)
  model, base_model = build_model()
  h1 = train_head(model, train_ds, val_ds)
  h2 = train_all(model, base_model, train_ds, val_ds, test_ds)

  return model

copy_data(Config.SRC_FACE ,Config.DST_FACE)
copy_data(Config.SRC_NO_FACE ,Config.DST_NO_FACE)
trained_model = main()
