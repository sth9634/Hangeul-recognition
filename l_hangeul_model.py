import argparse
import io
import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, './labels/2350-common-hangeul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'tfrecords-output')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'saved-model')

MODEL_NAME = 'hangeul_tensorflow'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

num_classes = 2350


def _parse_function(example):
    features = tf.io.parse_single_example(
        example,
        features={
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })
    label = features['image/class/label']
    image_encoded = features['image/encoded']

    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH*IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return image, label


def export_model(model_output_dir, input_node_names, output_node_name):
    """Export the model so we can use it later.

    This will create two Protocol Buffer files in the model output directory.
    These files represent a serialized version of our model with all the
    learned weights and biases. One of the ProtoBuf files is a version
    optimized for inference-only usage.
    """

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.GFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)


def weight_variable(shape):
    """Generates a weight variable of a given shape."""
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    """Generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias')


def main(label_file, tfrecords_dir, model_output_dir, num_train_epochs, num_target_accurate, batch_size):
    """Perform graph definition and model training.

    Here we will first create our input pipeline for reading in TFRecords
    files and producing random batches of images and labels.
    Next, a convolutional neural network is defined, and training is performed.
    After training, the model is exported to be used in applications.
    """
    global num_classes
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    # Define names so we can later reference specific nodes for when we use
    # the model for inference later.
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.io.gfile.glob(tf_record_pattern)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.io.gfile.glob(tf_record_pattern)

    # Create training dataset input pipeline.
    train_dataset = tf.data.TFRecordDataset(train_data_files) \
        .map(_parse_function) \
        .shuffle(100000) \
        .repeat(num_train_epochs) \
        .batch(batch_size) \
        .prefetch(1)

    # Create the model!

    # Placeholder to feed in image data.
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH*IMAGE_HEIGHT], name=input_node_name)
    # Placeholder to feed in label data. Labels are represented as one_hot
    # vectors.
    y_ = y_ = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

    # Reshape the image back into two dimensions so we can perform convolution.
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # First conv: reduce filters
    W_conv1 = weight_variable([5, 5, 1, 16])
    b_conv1 = bias_variable([16])
    x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv1 = tf.nn.relu(x_conv1 + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Second conv: reduce filters
    W_conv2 = weight_variable([5, 5, 16, 32])
    b_conv2 = bias_variable([32])
    x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv2 = tf.nn.relu(x_conv2 + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Third conv: optional (can be removed for even smaller model)
    W_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])
    x_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
                           padding='SAME')
    h_conv3 = tf.nn.relu(x_conv3 + b_conv3)
    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer (smaller)
    h_pool_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
    W_fc1 = weight_variable([8 * 8 * 64, 256])  # Reduce from 1024 to 256
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    # Dropout layer
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)

    # Output layer
    W_fc2 = weight_variable([256, num_classes])
    b_fc2 = bias_variable([num_classes])
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    tf.nn.softmax(y, name=output_node_name)
    
    # Define our loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(y_),
            logits=y
        )
    )

    # Define our optimizer for minimizing our loss. Here we choose a learning
    # rate of 0.0001 with AdamOptimizer. This utilizes someting
    # called the Adam algorithm, and utilizes adaptive learning rates and
    # momentum to get past saddle points.
    train_step = tf.train.AdamOptimizer(0.00005).minimize(cross_entropy)

    # Define accuracy.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')

        # Save the graph definition to a file.
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)

        try:
            iterator = train_dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            step = 0
            correct = 0

            while True:

                train_images, train_labels = sess.run(batch)

                sess.run(train_step, feed_dict={x: train_images,
                                                y_: train_labels,
                                                keep_prob: 0.5})
                if step % 500 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
                    print(f"Step {step}, Training Accuracy : {train_accuracy:.5f}")

                if train_accuracy == 1:
                    correct += 1
                    if correct >= num_target_accurate:
                        print(f"Finished training the model on Step {step}")
                        break

                if step % 10000 == 0:
                    saver.save(sess, checkpoint_file, global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            pass

        saver.save(sess, checkpoint_file)
        print('Testing model...')

        # Create testing dataset input pipeline.
        test_dataset = tf.data.TFRecordDataset(test_data_files) \
            .map(_parse_function) \
            .batch(batch_size) \
            .prefetch(1)

        accuracy2 = tf.reduce_sum(correct_prediction)
        total_correct_preds = 0
        total_preds = 0

        try:
            iterator = test_dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            while True:
                test_images, test_labels = sess.run(batch)
                acc = sess.run(accuracy2, feed_dict={x: test_images,
                                                     y_: test_labels,
                                                     keep_prob: 1.0})
                total_preds += len(test_images)
                total_correct_preds += acc

        except tf.errors.OutOfRangeError:
            pass

        test_accuracy = total_correct_preds/total_preds
        print(f"Testing Accuracy {test_accuracy:.5f}")

        export_model(model_output_dir, [input_node_name, keep_prob_node_name], output_node_name)

        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--epochs', type=int,
                        dest='num_train_epochs',
                        default=15,
                        help='Number of times to iterate over all of the training data.')
    parser.add_argument('--iteration', type=int,
                        dest='num_target_accurate',
                        default=3,
                        help='Number of times for getting all corect 128 answers.')
    parser.add_argument('--batch', type=int,
                        dest='batch_size',
                        default=128,
                        help='Number of batches')
    args = parser.parse_args()
    main(args.label_file, args.tfrecords_dir, args.output_dir, args.num_train_epochs, args.num_target_accurate, args.batch_size)
