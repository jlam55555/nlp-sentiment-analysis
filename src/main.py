import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub

import os
import pandas as pd
import json
import requests


# Path to scraped data
DATA_PATH = "data/hydrated/all.json"

# Constants
TRAIN_PERCENT = 80
NUM_EPOCHS = 3
BATCH_SIZE = 16
NUM_EXAMPLES_TO_VIEW = 100 // BATCH_SIZE    # Number of examples to display the model's prediction for

# Model Constants
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
# OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.0005) # Use with dropout
# OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.00025)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy()
METRICS = tf.keras.metrics.SparseCategoricalAccuracy()
AUTOTUNE = tf.data.AUTOTUNE

# Set the checkpoint
checkpoint_path = os.getcwd() + "/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights from:
# https://www.tensorflow.org/tutorials/keras/save_and_load
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)  # Saves every epoch


# From:
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


# The following 2 functions are from:
# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/solve_glue_tasks_using_bert_on_tpu.ipynb
def make_bert_preprocess_model(sentence_features, seq_length=128):

    """Returns Model mapping string features to BERT inputs.

    Args:
        sentence_features: a list with the names of string-valued features.
        seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
        A Keras Model that can be called on a list or dict of string Tensors
        (with the order or names, resp., given by sentence_features) and
        returns a dict of tensors for input to BERT.
    """

    input_segments = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load("http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in input_segments]

    # Optional: Trim segments in a smart way to fit seq_length.
    # Simple cases (like this example) can skip this step and let
    # the next step apply a default truncation to approximately equal lengths.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(bert_preprocess.bert_pack_inputs,
                            arguments=dict(seq_length=seq_length),
                            name='packer')
    model_inputs = packer(truncated_segments)
    return tf.keras.Model(input_segments, model_inputs)


def build_classification_model(num_features):

    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                                trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(512, activation=tf.keras.layers.ReLU(), name='finetune1')(net)
    net = tf.keras.layers.Dense(256, activation=tf.keras.layers.ReLU(), name='finetune2')(net)
    net = tf.keras.layers.Dense(128, activation=tf.keras.layers.ReLU(), name='finetune3')(net)
    net = tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU(), name='finetune4')(net)
    net = tf.keras.layers.Dense(num_features, activation=None, name='regression')(net)
    net = tf.keras.layers.Softmax()(net)
    return tf.keras.Model(inputs, net, name='prediction')


def parseData():

    # Read the json
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Parse the json
    parsed = json.loads(raw)
    parsed = [ ( str(x['id']), np.array( [x['text']] ), x['label2']+1 ) for x in parsed ]

    # Split the list of tuples into separate lists
    names, text, label = zip(*parsed)

    return list(names), list(text), list(label)


if __name__ == '__main__':

    # Load the data
    _, text, label = parseData()
    num_examples = len(label)
    print( "Number of examples:", num_examples )

    # Pre-process the data
    bert_preprocess_model = make_bert_preprocess_model(['sentence'])
    
    # Prepare the dataset
    print("Preparing the dataset...")
    dataset = tf.data.Dataset.from_tensor_slices( (text, label) )

    # Split the dataset
    # Idea from https://stackoverflow.com/a/58452268
    dsTrain = dataset.enumerate() \
                    .filter( lambda x,y: x % 100 < TRAIN_PERCENT ) \
                    .map( lambda x,y: y )
    dsTest = dataset.enumerate() \
                    .filter( lambda x,y: x % 100 >= TRAIN_PERCENT ) \
                    .map( lambda x,y: y )

    # Finish preparing the datasets
    num_train_examples = ( num_examples // 100 * TRAIN_PERCENT ) + ( num_examples % 100 
                                                                        if (num_examples % 100 < TRAIN_PERCENT) \
                                                                        else num_examples % 100 - TRAIN_PERCENT )
    num_test_examples = num_examples - num_train_examples
    print( "Number of training examples:", num_train_examples )
    print( "Number of testing examples:", num_test_examples )
    dsTrain = dsTrain \
                    .shuffle(num_train_examples) \
                    .repeat() \
                    .batch(BATCH_SIZE) \
                    .map( lambda ex,label: (bert_preprocess_model(ex),label) ) \
                    .cache().prefetch(buffer_size=AUTOTUNE)
    dsTest = dsTest \
                    .batch(BATCH_SIZE) \
                    .map( lambda ex,label: (bert_preprocess_model(ex),label) ) \
                    .cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    print("Building the model...")
    model = build_classification_model(3)

    # Drop the label b/c testing the model
    print("Starting testing...")

    # Setup the model
    steps_per_epoch = num_train_examples // BATCH_SIZE
    model.compile( optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS )

    # Test the initial performance of the model
    # model.evaluate(
    #     x=dsTest,
    #     steps=num_test_examples // BATCH_SIZE
    # )
    y_init = model.predict( dsTest.take(NUM_EXAMPLES_TO_VIEW) )
    print(y_init)

    # Train the model
    model.fit(
        x=dsTrain,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=dsTest,
        validation_steps=num_test_examples // BATCH_SIZE,
        callbacks=cp_callback
    )

    # Test the performance of the trained model
    model.evaluate(
        x=dsTest,
        steps=num_test_examples // BATCH_SIZE
    )
    y = model.predict( dsTest.take(NUM_EXAMPLES_TO_VIEW) )
    print(y)