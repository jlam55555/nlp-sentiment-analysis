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

# Model Constants
NUM_EPOCHS = 20
BATCH_SIZE = 16
# OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.0005)
LOSS = tf.keras.losses.MeanSquaredError()
METRICS = tf.keras.metrics.MeanSquaredError()
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
    period=50)  # Saves every 10 epochs


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


# Maybe add another dense layer
def build_regression_model(num_features):

    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                                trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(rate=0.1)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Dense(128, activation=tf.keras.layers.ReLU(), name='finetune1')(net)
    net = tf.keras.layers.Dense(64, activation=tf.keras.layers.ReLU(), name='finetune2')(net)
    net = tf.keras.layers.Dense(32, activation=tf.keras.layers.ReLU(), name='finetune3')(net)
    net = tf.keras.layers.Dense(num_features, activation=None, name='regression')(net)
    return tf.keras.Model(inputs, net, name='prediction')


def parseData():

    # Read the json
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Parse the json
    parsed = json.loads(raw)
    parsed = [ ( str(x['id']), np.array( [x['text']] ), x['label'] ) for x in parsed ]

    # Split the list of tuples into separate lists
    names, text, label = zip(*parsed)

    # Convert to a dict of tensors
    # inputs = { text), 'label': tf.convert_to_tensor(label) }

    return list(names), list(text), list(label)


if __name__ == '__main__':

    # Load the data
    _, text, label = parseData()
    num_examples = len(label)
    print(num_examples)
    label *= 100
    # Pre-process the data
    bert_preprocess_model = make_bert_preprocess_model(['sentence'])
    
    # Prepare the dataset
    print("Preparing the dataset...")
    dsTrain = tf.data.Dataset.from_tensor_slices( (text[:100], label[:100]) ) \
                    .shuffle(100) \
                    .repeat() \
                    .batch(BATCH_SIZE) \
                    .map( lambda ex,label: (bert_preprocess_model(ex),label) ) \
                    .cache().prefetch(buffer_size=AUTOTUNE)

                    #(1,)       sentences
                    #(N,)       batching
                    #(N,128)    embedding

                    #(1,)       sentences
                    #(1,128)    embedding
                    #(N,1,128)  batching

    ################################ CHANGE SUFFLE TO NUM_TRAIN AND NUM_TEST
    # Split the dataset
    # Idea from https://stackoverflow.com/a/58452268
    # dsTrain = dataset.enumerate() \
    #                 .filter( lambda x,y: x % 100 < TRAIN_PERCENT ) \
    #                 .map( lambda x,y: y )
    # dsTest = dataset.enumerate() \
    #                 .filter( lambda x,y: x % 100 >= TRAIN_PERCENT ) \
    #                 .map( lambda x,y: y )
    
    # Finish preparing the datasets
    # dsTrain = dsTrain.shuffle(num_examples) \
    #                 .repeat() \
    #                 .batch(BATCH_SIZE) \
    #                 .cache().prefetch(buffer_size=AUTOTUNE)
    # dsTest = dsTest.batch(BATCH_SIZE) \
    #                 .cache().prefetch(buffer_size=AUTOTUNE)

    # Build the model
    print("Building the model...")
    regression_model = build_regression_model(1)

    # Drop the label b/c testing the model
    print("Starting testing...")
    # for row in dataset.take(100).map(lambda x,y: x):
    #     bert_raw_result = regression_model(row)
    #     print(tf.sigmoid(bert_raw_result))

    # TODO: Train the model
    steps_per_epoch = 100 // BATCH_SIZE
    regression_model.compile( optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS )

    dsTest = tf.data.Dataset.from_tensor_slices( (text[100:150], label[100:150]) ) \
                    .batch(BATCH_SIZE) \
                    .map( lambda ex,label: (bert_preprocess_model(ex),label) )
    regression_model.evaluate(dsTest)
    y_init = regression_model.predict(dsTest)

    regression_model.fit(
        x=dsTrain,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks = cp_callback
    )

    # dsTest = tf.data.Dataset.from_tensor_slices( (text[100:150], label[100:150]) ) \
    #                 .batch(BATCH_SIZE) \
    #                 .map( lambda ex,label: (bert_preprocess_model(ex),label) )
    regression_model.evaluate(dsTest)
    y = regression_model.predict(dsTest)
    df = pd.DataFrame( list(zip(y_init, y, label[100:150])), columns=['Init', 'Predicted', 'Actual'] )
    print(df)