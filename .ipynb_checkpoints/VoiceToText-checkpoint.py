import os
import tensorflow as tf
import numpy as np
import re

# Speech recognition
import speech_recognition as sr
import pyaudio

# ---------------------------------------------------
# 1) SPEECH RECOGNITION: Capture Spanish speech
# ---------------------------------------------------
def capture_spanish_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, one moment...")
        r.adjust_for_ambient_noise(source, duration=1)

        print("Por favor, hable en español ahora:")
        audio_data = r.listen(source)

    try:
        # Recognize Spanish speech using Google's online API
        text_spanish = r.recognize_google(audio_data, language="es-ES")
        print("Captured Spanish text:", text_spanish)
        return text_spanish
    except sr.UnknownValueError:
        print("Lo siento, no pude entender el audio.")
        return None
    except sr.RequestError as e:
        print(f"Error al solicitar resultados al servicio de Google: {e}")
        return None

# ---------------------------------------------------
# 2) DOWNLOAD & PREPARE THE SPANISH–ENGLISH DATASET
# ---------------------------------------------------
def load_spa_eng_dataset():
    # Download the dataset file (spa-eng.zip) and extract
    zip_file = tf.keras.utils.get_file(
        'spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True
    )
    # The extracted text file
    file_path = os.path.join(os.path.dirname(zip_file), "spa-eng", "spa.txt")

    # Each line in spa.txt is "eng [tab] spa"
    # But the version you found might be "spa [tab] eng" or "eng [tab] spa"
    # We need to confirm the order. According to the official spa-eng, lines are typically "Go.    Ve."
    # i.e. English <TAB> Spanish. We'll parse accordingly.
    # We'll assume: eng\tspa
    spa_sentences = []
    eng_sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        eng, spa = line.split('\t')[:2]
        spa_sentences.append(spa)
        eng_sentences.append(eng)

    return spa_sentences, eng_sentences

# Simple cleaner (optional, remove punctuation etc.)
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # Insert spaces before/after punctuation (rough approach)
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" ]+', " ", sentence)
    sentence = sentence.strip()
    return sentence

# ---------------------------------------------------
# 3) TOKENIZE & BUILD SEQ2SEQ MODEL
# ---------------------------------------------------
def build_tokenizers_and_model(spa_sentences, eng_sentences, vocab_size=8000):
    # Preprocess
    spa_sentences = [preprocess_sentence(s) for s in spa_sentences]
    eng_sentences = [preprocess_sentence(s) for s in eng_sentences]

    # We'll add start/end tokens for the target language
    # For a real seq2seq, we typically add them for the source or both, but keep it simple
    eng_sentences_in = ["<start> " + s + " <end>" for s in eng_sentences]

    # Tokenizers
    spa_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',  # We already handled basic punctuation
        num_words=vocab_size,
        oov_token="<unk>"
    )
    eng_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        num_words=vocab_size,
        oov_token="<unk>"
    )

    spa_tokenizer.fit_on_texts(spa_sentences)
    eng_tokenizer.fit_on_texts(eng_sentences_in)

    # Convert text -> integer sequences
    spa_tensor = spa_tokenizer.texts_to_sequences(spa_sentences)
    eng_tensor = eng_tokenizer.texts_to_sequences(eng_sentences_in)

    # Pad sequences to have the same length
    spa_tensor = tf.keras.preprocessing.sequence.pad_sequences(spa_tensor, padding='post')
    eng_tensor = tf.keras.preprocessing.sequence.pad_sequences(eng_tensor, padding='post')

    # Build a small seq2seq (Embedding + GRU)
    # We'll embed Spanish (encoder) -> single GRU -> decode with another GRU.
    embedding_dim = 256
    units = 256

    # Input vocab sizes
    input_vocab_size = len(spa_tokenizer.word_index) + 1
    output_vocab_size = len(eng_tokenizer.word_index) + 1

    # ---------------------
    # ENCODER
    # ---------------------
    encoder_inputs = tf.keras.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h = tf.keras.layers.GRU(units, return_state=True)(encoder_embedding)

    # ---------------------
    # DECODER
    # ---------------------
    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    # We feed the encoder state as initial_state
    decoder_outputs, _ = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)(
        decoder_embedding, initial_state=state_h
    )
    # Output layer: project to the size of the output vocab
    decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return spa_tokenizer, eng_tokenizer, spa_tensor, eng_tensor, model

# ---------------------------------------------------
# 4) TRAIN & INFER
# ---------------------------------------------------
def train_seq2seq(spa_tensor, eng_tensor, model, batch_size=64, epochs=3):
    # For the decoder, we shift targets by one.
    # Typically you'd create eng_input and eng_target (shifted by 1) for teacher forcing.
    # For simplicity, we’ll assume we feed the same sequences in as input & shift inside the model or we
    # can do a quick approach: eng_in = eng_tensor[:,:-1], eng_out = eng_tensor[:,1:].
    eng_in = eng_tensor[:, :-1]
    eng_out = eng_tensor[:, 1:]

    history = model.fit(
        [spa_tensor, eng_in],
        tf.expand_dims(eng_out, -1),  # Must match output shape (batch, seq, 1)
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )
    return history

def translate_spa_to_eng(input_sentence, spa_tokenizer, eng_tokenizer, model):
    """Use the trained encoder-decoder model to infer a translation for a single Spanish sentence."""
    # Preprocess
    input_sentence = preprocess_sentence(input_sentence)
    # Convert to sequence
    input_seq = spa_tokenizer.texts_to_sequences([input_sentence])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=None, padding='post')

    # Encode
    encoder_model = tf.keras.Model(model.input[0], model.layers[3].output)  # up to GRU state_h
    state_h = encoder_model.predict(input_seq)

    # Build a minimal decoder (one step at a time)
    dec_embedding_layer = model.layers[4]
    dec_gru_layer = model.layers[5]
    dec_dense_layer = model.layers[6]

    # Start token for English
    start_token = eng_tokenizer.word_index.get('<start>')
    end_token   = eng_tokenizer.word_index.get('<end>')

    dec_input = tf.expand_dims([start_token], 0)  # shape (1,1)
    result_tokens = []

    # We'll set an arbitrary max steps
    for _ in range(50):
        dec_embed = dec_embedding_layer(dec_input)
        dec_output, dec_state = dec_gru_layer(dec_embed, initial_state=state_h)
        preds = dec_dense_layer(dec_output)
        preds_id = tf.argmax(preds[0, -1, :]).numpy()

        if preds_id == 0 or preds_id == end_token:
            break

        # convert id->word
        result_tokens.append(preds_id)
        # next input
        dec_input = tf.expand_dims([preds_id], 0)
        state_h = dec_state

    # Convert predicted tokens back to words
    inv_eng_index = {v:k for k,v in eng_tokenizer.word_index.items()}
    predicted_words = [inv_eng_index.get(idx, '') for idx in result_tokens]
    return ' '.join(predicted_words)

# ---------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------
def main():
    # 1) Capture Spanish speech
    spanish_text = capture_spanish_speech()
    if not spanish_text:
        return

    # 2) Load dataset (Spanish–English)
    print("Loading the Spanish–English dataset...")
    spa_sentences, eng_sentences = load_spa_eng_dataset()

    # 3) Build model & train
    print("Building tokenizers and seq2seq model...")
    spa_tok, eng_tok, spa_tensor, eng_tensor, model = build_tokenizers_and_model(spa_sentences, eng_sentences)

    print("Training the model on a subset of the dataset (this may take a while)...")
    train_seq2seq(spa_tensor[:20000], eng_tensor[:20000], model, epochs=3)
    # Using only first 20k lines to speed up. Increase or remove slice for better results.

    # 4) Translate recognized Spanish text to English
    translation = translate_spa_to_eng(spanish_text, spa_tok, eng_tok, model)
    print("Translated to English:", translation)

if __name__ == "__main__":
    main()
