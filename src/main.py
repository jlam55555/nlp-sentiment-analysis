import json
import requests
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub
import numpy as np


# Path to scraped data
DATA_PATH = "data/hydrated/all.json"

# Constants
TRAIN_PERCENT = 80

# Model Constants
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.MeanSquaredError()
METRICS = tf.keras.metrics.MeanSquaredError()

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
def build_regression_model(num_classes):

    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                                trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(rate=0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='regression')(net)
    return tf.keras.Model(inputs, net, name='prediction')


def parseData():

    # Read the json
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Parse the json
    parsed = json.loads(raw)
    parsed = [ ( str(x['id']), { 'sentence': tf.reshape(tf.convert_to_tensor(x['text']), [1]), 
                                'label': tf.reshape(tf.convert_to_tensor(x['label']), [1]) } ) for x in parsed ]

    # Split the list of tuples into separate lists
    names, inputs = zip(*parsed)

    # Convert to a dict of tensors
    # inputs = { text), 'label': tf.convert_to_tensor(label) }

    return list(names), inputs


if __name__ == '__main__':

    # Load the data
    _, inputs = parseData()
    # # print(inputs)
    # for ex in inputs:
    #     print(ex)
    num_examples = len(inputs)
    print(num_examples)

    # Pre-process the data
    bert_preprocess_model = make_bert_preprocess_model(['sentence'])
    
    # Prepare the dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs) \
                .map(lambda ex: (bert_preprocess_model(ex), ex['label'])) \
                .shuffle(num_examples, reshuffle_each_iteration=False) \

    for x,y in dataset.enumerate().take(2):
        print( x,y )

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
    regression_model = build_regression_model(1)

    # Drop the label b/c testing the model
    for row in dataset.take(5).map(lambda x,y: x):
        bert_raw_result = regression_model(row)
        print(tf.sigmoid(bert_raw_result))

    # TODO: Train the model
    regression_model.compile( optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS )
        