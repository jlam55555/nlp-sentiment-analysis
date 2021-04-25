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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


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

    docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["It's a trap!"]])
    tokenizer = tf_text.WhitespaceTokenizer()
    tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))

    iterator = iter(tokenized_docs)
    print(next(iterator).to_list())
    print(next(iterator).to_list())
    
    # TODO: ???

    pass