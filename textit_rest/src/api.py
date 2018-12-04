from __future__ import print_function
from flask import Flask
from flask import request
from flask import json
from flask import jsonify
from flask_cors import CORS
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input, LSTM, Dense
import numpy as np
from keras import metrics
import copy
import pickle
from nltk.translate.bleu_score import sentence_bleu

epochs = 5  # Number of epochs to train for.

encoder_model=[]
decoder_model=[]
num_decoder_tokens=0
max_decoder_seq_length=0
target_token_index=[]
reverse_target_char_index=[]
input_token_index=[]
encoder_input_data=[]
input_texts=[]
model_loaded=False
api_directory="C:\\Users\\USER\\PycharmProjects\\textit_rest\\"

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    print('/')
    return "Welcome To TextIt!"

@app.route('/get_common_list')
def getCommonList():
    data = json.load(open(api_directory+"data\\list.json"))
    return jsonify(data)

@app.route('/get_twitter_list')
def getTwitterList():
    data = json.load(open(api_directory+"data\\twitterList.json"))
    return jsonify(data)


@app.route('/translate_new_sentence',methods=['POST'])
def translate_new_sentence():
    load_model()
    sentence = request.json['sentence'].lower()
    copy_encoder_input_data = copy.deepcopy(encoder_input_data)
    copy_encoder_input_data[copy_encoder_input_data == 1] = 0
    print('----')
    for t, char in enumerate(sentence):
        copy_encoder_input_data[0, t, input_token_index[char]] = 1.
    copy_input_seq = copy_encoder_input_data[0:1]
    copy_decoded_sentence = decode_sequence(copy_input_seq)
    print('sentence: ', sentence)
    print('translate: ', copy_decoded_sentence)
    return copy_decoded_sentence


@app.route('/load_model')
def load_model():
    global input_token_index
    global encoder_input_data
    global encoder_model
    global decoder_model
    global num_decoder_tokens
    global max_decoder_seq_length
    global target_token_index
    global reverse_target_char_index
    global model_loaded
    if model_loaded==False:
        directory_path=api_directory+"\\trained_model"
        input_token_index = pickle.load(open(directory_path+"\\input_token_index", "rb"))
        encoder_input_data = pickle.load(open(directory_path+"\\encoder_input_data", "rb"))
        json_file = open(directory_path+'\\encoder_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        encoder_model = model_from_json(loaded_model_json)
        encoder_model.load_weights(directory_path+"\\encoder_model.h5")
        json_file = open(directory_path+'\\decoder_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        decoder_model = model_from_json(loaded_model_json)
        decoder_model.load_weights(directory_path+"\\decoder_model.h5")
        num_decoder_tokens = pickle.load(open(directory_path+"\\num_decoder_tokens", "rb"))
        target_token_index = pickle.load(open(directory_path+"\\target_token_index", "rb"))
        reverse_target_char_index = pickle.load(open(directory_path+"\\reverse_target_char_index", "rb"))
        max_decoder_seq_length = pickle.load(open(directory_path+"\\max_decoder_seq_length", "rb"))
        model_loaded=True


#dont forget to run bleu score on the test set of the file that was learned from
@app.route('/run_bleu_score')
def run_bleu_score():
    load_model()
    data_path = api_directory+"\\data\\20percent_sen8.txt"
    input_texts = []
    target_texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().lower().split('\n')
    for line in lines[:  len(lines) - 1]:
        input_text, target_text = line.split('\t')
        input_texts.append(input_text)
        target_texts.append(target_text)

    sum_score = 0

    for seq_index in range(len(input_texts)):
        sentence=input_texts[seq_index]
        copy_encoder_input_data = copy.deepcopy(encoder_input_data)
        copy_encoder_input_data[copy_encoder_input_data == 1] = 0
        #print('----')
        for t, char in enumerate(sentence):
            copy_encoder_input_data[0, t, input_token_index[char]] = 1.
        copy_input_seq = copy_encoder_input_data[0:1]
        copy_decoded_sentence = decode_sequence(copy_input_seq)
        #print('sentence: ', sentence)
        #print('translate: ', copy_decoded_sentence)

        reference=target_texts[seq_index].split()
        if len(reference)==3:
            sum_score = sum_score + sentence_bleu([reference],copy_decoded_sentence.split(),weights=(1/3,1/3,1/3))
        elif len(reference)==2:
            sum_score = sum_score + sentence_bleu([reference],copy_decoded_sentence.split(),weights=(1/2,1/2))
        elif len(reference)==1:
            sum_score = sum_score + sentence_bleu([reference],copy_decoded_sentence.split(),weights=(1,0))
        else:
            sum_score = sum_score + sentence_bleu([reference],copy_decoded_sentence.split())
    avg_score = sum_score / (len(input_texts))
    print(avg_score)
    return str(avg_score)



@app.route('/run_bleu_score_for_tranl8it')
def run_bleu_score_for_tranl8it():
    load_model()
    data_path = api_directory+"\\data\\sen5cut25sentences.txt"
    input_texts = []
    target_texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().lower().split('\n')
    for line in lines[:  len(lines) - 1]:
        input_text, target_text = line.split('\t')
        input_texts.append(input_text)
        target_texts.append(target_text)

    data_path = api_directory+"\\data\\transl8itOutput.txt"
    transl8it_texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().lower().split('\n')
    for line in lines[:  len(lines) - 1]:
        input_text, target_text = line.split('\t')
        transl8it_texts.append(target_text)

    sum_score = 0

    for seq_index in range(len(input_texts)):
        sentence=input_texts[seq_index]
        translated_sentence=transl8it_texts[seq_index]
        print('sentence: ', sentence)
        print('translate: ', translated_sentence)

        reference=target_texts[seq_index].split()
        if len(reference)==3:
            sum_score = sum_score + sentence_bleu([reference],translated_sentence.split(),weights=(1/3,1/3,1/3))
        elif len(reference)==2:
            sum_score = sum_score + sentence_bleu([reference],translated_sentence.split(),weights=(1/2,1/2))
        elif len(reference)==1:
            sum_score = sum_score + sentence_bleu([reference],translated_sentence.split(),weights=(1,0))
        else:
            sum_score = sum_score + sentence_bleu([reference],translated_sentence.split())
    avg_score = sum_score / (len(input_texts))
    print(avg_score)
    return str(avg_score)


#not the most updated. the most updated is in a different project (seq2seq)
@app.route('/learn')
def learn():
    global epochs
    batch_size = 64  # Batch size for training.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 10000  # Number of samples to train on.
    # Path to the data txt file on disk.
    data_path = api_directory+"\\data\\sen4.txt"

    # Vectorize the data.
    global input_texts
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().lower().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    global num_decoder_tokens
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    global max_decoder_seq_length
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    global input_token_index
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    global target_token_index
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    global encoder_input_data
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save('s2s.h5')
    global encoder_model
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    global decoder_model
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    global reverse_target_char_index
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())

    return ""


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

@app.route('/reset_model')
def reset_model():
    global encoder_model, decoder_model, num_decoder_tokens, num_decoder_tokens, max_decoder_seq_length, target_token_index, target_token_index, reverse_target_char_index,input_token_index, encoder_input_data, input_texts
    encoder_model = None
    decoder_model = None
    num_decoder_tokens = 0
    max_decoder_seq_length = 0
    target_token_index = None
    reverse_target_char_index = None
    input_token_index = None
    encoder_input_data = None
    input_texts = None
    return "reset_model"


if __name__ == '__main__':
    app.run(debug=True,port=4000)