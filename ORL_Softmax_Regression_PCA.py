import io
import itertools
import keras.callbacks
import scipy.io as sio
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.keras.applications.densenet import layers
from sklearn.metrics import confusion_matrix

# 设定初始参数
num_epoch = 200
learning_rate = 0.001

# 导入数据集
data = open('./datasets/ORL_64x64.mat', 'rb')
dict = sio.loadmat(data)
feature = dict['fea']
label = dict['gnd']

def pca(x, dim):
    '''
        x:输入矩阵
        dim:降维之后的维度数
    '''
    with tf.name_scope("PCA"):
        row, col = np.shape(x)
        mean = np.mean(x, axis=0)
        x_new = x - np.tile(mean, (row, 1))

        cov_x = np.cov(x_new.T)

        e, v = np.linalg.eig(cov_x)
        e = np.real(e)
        v = np.real(v)
        evs = v[:, :dim]
        pca = np.dot(x, evs)
        reshape = np.dot(pca, evs.T) * 255.0
    return pca, reshape

feature, reshape = pca(feature, dim=35)
print(feature.shape)

# 划分训练集和测试集
#  随机打乱
np.random.seed(100)
feature = np.random.permutation(feature)
np.random.seed(100)
label = np.random.permutation(label)

# 划分 80%(320)训练； 20%(80)测试
train_data = feature[0:320]
train_label = label[0:320]

test_data = feature[320:]
test_label = label[320:]

# 重构数据集
train_data = train_data.astype(np.float32) / 255.0
for i in range(0, train_label.shape[0]):
    train_label[i] -= 1

test_data = test_data.astype(np.float32) / 255.0
for i in range(0, test_label.shape[0]):
    test_label[i] -= 1

# 记录器
class_names = np.arange(0, 40, 1)
class_names = np.array(class_names)
logdir = './tensorboard_ORL_Softmax_Regression_PCA'

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
summary_writer = tf.summary.create_file_writer(logdir)

# 图片打印方法
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid():
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(1, 26):
        # Start next subplot.
        plt.subplot(5, 5, i, title=class_names[train_label[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        images = np.reshape(reshape[i], (64, 64))
        images = images.T
        plt.imshow(images, cmap=plt.cm.gray)

    return figure


# Confusion Matrix 生成方法
def plot_confusion_matrix(cm, class_names):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(16, 16))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


# Prepare the plot
figure = image_grid()
# Convert to image and log
with summary_writer.as_default():
    tf.summary.image("Training data", plot_to_image(figure), step=0)


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(train_data)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = confusion_matrix(train_label, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with summary_writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


# Define the per-epoch callback.
cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

callback = [
    keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
    tensorboard_callback,
    cm_callback
]


def softmax_regression():
    model = tf.keras.models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=40, activation=tf.nn.softmax))

    # 编译模型
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model


model = softmax_regression()
history = model.fit(train_data, train_label, epochs=num_epoch, callbacks=callback,
                    validation_data=(test_data, test_label))
model.summary()
print("===模型测试===")
model.evaluate(test_data, test_label)
