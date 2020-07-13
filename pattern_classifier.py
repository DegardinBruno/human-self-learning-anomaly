import tensorflow as tf
from tensorflow import keras
import numpy as np, os
from sklearn.metrics import roc_auc_score
import bayesian_classifier as BC


def create_train_dataset(xs, ys):
    return tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(ys)).batch(64)


def create_val_dataset(xs):
    return tf.data.Dataset.from_tensor_slices((xs)).batch(64)


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=192, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer='adagrad',
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


def train(y_train, x_train, aton_iteration, features_video):


    x_train = np.array([x_train[i: i+features_video] for i in range(0, len(x_train), features_video)])                     # Stack scores
    y_train = np.array([1 if max(y_train[i: i+features_video])>0 else 0 for i in range(0, len(y_train), features_video)])  # Stack labels


    train_dataset = create_train_dataset(x_train, y_train)
    model = create_model()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('models/pattern_model',str(aton_iteration),
                                                              'pattern_' + str(aton_iteration) + '.ckpt'),
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(
        train_dataset.repeat(),
        epochs=100,
        steps_per_epoch=100,
        callbacks=[cp_callback])


def test(x_test, test_val_flag, aton_iteration, features_video, y_test=None):

    x_test = [x_test[i: i+features_video] for i in range(0, len(x_test), features_video)]
    y_test = [1 if max(y_test[i: i+features_video])>0 else 0 for i in range(0, len(y_test), features_video)] if test_val_flag else None

    model = create_model()
    model.load_weights(os.path.join('models/pattern_model', str(aton_iteration), 'pattern_'+str(aton_iteration)+'.ckpt'))

    val_dataset = create_val_dataset(x_test)
    predictions = model.predict(val_dataset)


    BC.histogram(y_test, predictions[:, 0], 'pattern/VAL', aton_iteration) if test_val_flag else None
    if test_val_flag:
        classes, probs, max_neg_threshold, max_pos_threshold = BC.gaussian_kde(y_test, predictions[:, 0], 'pattern', aton_iteration)

    AUC = roc_auc_score(np.array(y_test), predictions[:, 0]) if test_val_flag else None
    print(AUC) if test_val_flag else None

    return np.array([0,1]), predictions





