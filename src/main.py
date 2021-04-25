import requests
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_hub as hub
import numpy as np


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


# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/solve_glue_tasks_using_bert_on_tpu.ipynb#scrollTo=KeHEYKXGqjAZ
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


AUTOTUNE = tf.data.AUTOTUNE


def load_dataset_from_tfds(in_memory_ds, info, split, batch_size,
                           bert_preprocess_model):
    is_training = split.startswith('train')
    dataset = tf.data.Dataset.from_tensor_slices(in_memory_ds[split])
    num_examples = info.splits[split].num_examples

    if is_training:
        dataset = dataset.shuffle(num_examples)
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ex: (bert_preprocess_model(ex), ex['label']))
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    return dataset, num_examples


def build_classifier_model(num_classes):
    inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(None,), dtype=tf.int32),
    )

    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
                                trainable=True, name='encoder')
    net = encoder(inputs)['pooled_output']
    net = tf.keras.layers.Dropout(rate=0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
    return tf.keras.Model(inputs, net, name='prediction')

if __name__ == '__main__':
    # TODO: grab dataset

    # TODO: preprocess
    # https://www.tensorflow.org/tutorials/tensorflow_text/tokenizers
    # tokenizer = tf_text.WhitespaceTokenizer()
    # tokens = tokenizer.tokenize(["What you know you can't explain, but you feel it."])
    # print(tokens.to_list())

    # Split into subtokens
    # url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
    # f = requests.get(url)
    # filepath = "vocab.txt"
    # open(filepath, 'wb').write(f.content)

    # subtokenizer = tf_text.UnicodeScriptTokenizer(filepath)
    # subtokens = tokenizer.tokenize(tokens)
    # print(subtokens.to_list())

    # docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
    # tokenizer = tf_text.WhitespaceTokenizer()
    # tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))

    # iterator = iter(tokenized_docs)
    # print(next(iterator).to_list())
    # print(next(iterator).to_list())

    # preprocessor = hub.load(
    #     "http://tfhub.dev/tensorflow/albert_en_preprocess/3")
    
    # # https://www.tensorflow.org/hub/common_saved_model_apis/text#text_embeddings_with_transformer_encoders
    # # Tokenize batches of both text inputs.
    # text_premises = tf.constant(["The quick brown fox jumped over the lazy dog.",
    #                          "Good day."])
    # tokenized_premises = preprocessor.tokenize(text_premises)
    # text_hypotheses = tf.constant(["The dog was lazy.",  # Implied.
    #                             "Axe handle!"])       # Not implied.
    # tokenized_hypotheses = preprocessor.tokenize(text_hypotheses)

    # # Pack input sequences for the Transformer encoder.
    # seq_length = 128
    # encoder_inputs = preprocessor.bert_pack_inputs(
    #     [tokenized_premises, tokenized_hypotheses],
    #     seq_length=seq_length)  # Optional argument.

    # print( encoder_inputs )

    # # Test with a tfds
    # ds_premises = tf.data.Dataset.from_tensor_slices( [ ["The quick brown fox jumped over the lazy dog.","Good day."],
    #                                                     ["The quick brown fox jumped over the lazy dog.","Good day."],
    #                                                     ["The quick brown fox jumped over the lazy dog.","Good day."],
    #                                                     ["The quick brown fox jumped over the lazy dog.","Good day."]] ) \
    #                                 .map( preprocessor.tokenize ) \
    #                                 .batch(3)
    # ds_hypotheses = tf.data.Dataset.from_tensor_slices( [["The dog was lazy.", "Axe handle!"]] ).map( preprocessor.tokenize )

    # ds_premises = ds_premises.map( preprocessor.bert_pack_inputs )

    # for sentence in ds_premises:
    #     print(sentence)

    # TODO: ???

    test_preprocess_model = make_bert_preprocess_model(['my_input1', 'my_input2'])
    test_text = [np.array(['some random test sentence']),
                np.array(['another sentence'])]
    text_preprocessed = test_preprocess_model(test_text)

    print('Keys           : ', list(text_preprocessed.keys()))
    print('Shape Word Ids : ', text_preprocessed['input_word_ids'].shape)
    print('Word Ids       : ', text_preprocessed['input_word_ids'][0, :16])
    print('Shape Mask     : ', text_preprocessed['input_mask'].shape)
    print('Input Mask     : ', text_preprocessed['input_mask'][0, :16])
    print('Shape Type Ids : ', text_preprocessed['input_type_ids'].shape)
    print('Type Ids       : ', text_preprocessed['input_type_ids'][0, :16])

    test_classifier_model = build_classifier_model(2)
    bert_raw_result = test_classifier_model(text_preprocessed)
    print(tf.sigmoid(bert_raw_result))