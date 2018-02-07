from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN"""
    # input layer
    input_layer = tf.reshape(features["x"], [-1, 100, 1])

    # convolutional layer #1
    conv1 = tf.layers.conv1d(
        inputs = input_layer,
        filters = 32,
        kernel_size = 10,
        padding = "same",
        activation = tf.nn.relu
    )

    # pooling layer #1
    pool1 = tf.layers.max_pooling1d(
        inputs = conv1,
        pool_size = 2,
        strides = 2
    )

    # convolutional layer #2
    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=64,
        kernel_size=10,
        padding="same",
        activation=tf.nn.relu
    )

    # pooling layer #2
    pool2 = tf.layers.max_pooling1d(
        inputs=conv2,
        pool_size=2,
        strides=2
    )

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 25 * 64])
    dense = tf.layers.dense(
        inputs = pool2_flat,
        units = 1024,
        activation = tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode==tf.estimator.ModeKeys.TRAIN
    )

    # logits layer
    logits = tf.layers.dense(
        inputs = dropout,
        units = 4
    )

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # configure the training op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )

"""
Return array of as many slice_size sized 1D arrays as can be sliced from sample file identified by ton_name.
"""
def get_sliced_up_sample(ton_name, slice_size):
    data = np.loadtxt('samples/' + ton_name + '.ton', dtype=np.float32)
    slice_number = int(len(data)/slice_size)
    data = np.split(data, slice_size*np.arange(1,slice_number))
    return np.array(data[:-1]) # throw away last slice because it might be too short

"""
Return whole sample file identified by ton_name as 1D array.
"""
def get_sample(ton_name):
    data = np.loadtxt('samples/' + ton_name + '.ton', dtype=np.float32)
    return data

def main(unused_argv):
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    learn_tones = ['E2', 'G2', 'D#3', 'silence']  # position indices in this list determine class labels
    sample_data = [get_sliced_up_sample(tn, 100) for tn in learn_tones]
    for k, s in enumerate(sample_data):
        np.random.shuffle(sample_data[k])
    #     print(len(s))
    # return 0
    train_len = 400
    train_data = np.concatenate([s[:train_len] for s in sample_data])
    train_labels = np.concatenate([np.ones(train_len, dtype=np.int32) for s in sample_data])
    eval_data = np.concatenate([s[train_len:] for s in sample_data])
    eval_labels = np.concatenate([np.ones(len(s)-train_len, dtype=np.int32) for s in sample_data])


    # create the Estimator
    bass_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/bass_convnet_model"
    )

    # set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": train_data},
        y = train_labels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )
    bass_classifier.train(
        input_fn = train_input_fn,
        steps = 2000,
        hooks = [logging_hook]
    )

    # evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": eval_data},
        y = eval_labels,
        num_epochs = 1,
        shuffle = False
    )
    eval_results = bass_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()