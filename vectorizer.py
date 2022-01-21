
import re
import string
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from setup import get_pairs

strip_chars = (string.punctuation).replace("[", "").replace("]", "")
vocab_size = 15000
sequence_length = 20
batch_size = 64

train_pairs, val_pairs, test_pairs = get_pairs()

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def vectorizer(tokens = 'input'):
    seq = sequence_length if tokens == 'input' else (sequence_length + 1)
    std = 'lower_and_strip_punctuation' if tokens == 'input' else custom_standardization
    start_idx = 0 if tokens == 'input' else 1
    vect_obj = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length = seq,
        standardize=std)
    train_texts = [pair[start_idx] for pair in train_pairs]
    vect_obj.adapt(train_texts)
    return vect_obj

eng_vectorization = vectorizer()
ind_vectorization = vectorizer('output')

def format_dataset(eng, ind):
    eng = eng_vectorization(eng)
    ind = ind_vectorization(ind)
    return ({"encoder_inputs": eng, "decoder_inputs": ind[:, :-1],}, ind[:, 1:])

def make_dataset(pairs):
    eng_texts, ind_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    ind_texts = list(ind_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, ind_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

