# Decoding with existing `model.h5`.
# Alternatively, retrain by running model.py -- freeze layers if you want :)

import tensorflow as tf, numpy as np, random
from vectorizer import ind_vectorization, eng_vectorization, test_pairs
model = tf.keras.models.load_model('new_model.h5')

ind_vocab = ind_vectorization.get_vocabulary()
ind_index_lookup = dict(zip(range(len(ind_vocab)), ind_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    tokenized_input_sentence = np.expand_dims(tokenized_input_sentence, 1)
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = ind_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = ind_index_lookup[sampled_token_index]
        if sampled_token == "end":
            break
        decoded_sentence += " " + sampled_token
    return decoded_sentence + " [end]"

def get_random_inputs(n_inputs = 30):
    test_input = [pair[0] for pair in test_pairs]
    for _ in range(30):
        input = random.choice(test_input)
        print("Input:" + random.choice(input))
        print("Translated: " + decode_sequence(input))

decode_sequence("Hello")