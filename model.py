import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import vectorizer, typing
from transformer import PositionalEmbedding, TransformerDecoder, TransformerEncoder
from vectorizer import sequence_length, vocab_size

train_ds = vectorizer.make_dataset(vectorizer.train_pairs)
val_ds = vectorizer.make_dataset(vectorizer.val_pairs)

es = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    mode='max', 
    verbose=1, 
    patience=10)
# Define model checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(
    "model_checkpoints", 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode="max")

def build_model(embed_dim = 256, latent_dim = 256, num_heads = 8):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = layers.Dropout(0.2)(x)
    decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
    transformer.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return transformer

def plot_model(model, dpi = 200, filename = 'model.png'):
    keras.utils.plot_model(model, to_file=filename, show_shapes=True, dpi=dpi)

def fit_model(model, epochs=50, verbose = 1, callbacks = None):
    model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks = callbacks)

def export_model(model, path):
    model.save(path)

if __name__ == "__main__":
    model = build_model()
    print(model.summary())
    fit_model(model, callbacks = [es])
    export_model(model, 'new_model.h5')