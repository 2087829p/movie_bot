# Building a ChatBot with Deep NLP

import numpy as np
import tensorflow as tf
import re
import time
import ast
from collections import Counter

#### PART 1 - DATA PREPROCESSING #####

#lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# dict that maps each line to its id
id2line = {}
with open('movie_lines.txt', encoding='utf-8', errors='ignore') as f:
    for line in f:
        l = line.split(' +++$+++ ')
        if len(l) == 5:
            id2line[l[0]] = l[-1]
            

# list of convos
conversations_ids = []
q_words = Counter()
a_words = Counter()
for c in conversations[:-1]:
    cl = ast.literal_eval(c.split(' +++$+++ ')[-1])
    conversations_ids.append(cl.split(','))

del conversations
questions = []
answers = []
for c in conversations_ids:
    for i in range(len(c) - 1):
        q = clean_text(id2line[c[i]])
        questions.append(q)
        q_words.update(q.split())
        a = clean_text(id2line[c[i+1]])
        answers.append(a)
        a_words.update(a.split())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

# map q words and a words to unique ids
threshhold = 20
qwords2int = {}
word_number = 0
for w in q_words:
    if q_words[w] >= threshhold:
        qwords2int[w] = word_number
        word_number += 1
awords2int = {} 
word_number = 0
for w in a_words:
    if a_words[w] >= threshhold:
        awords2int[w] = word_number
        word_number += 1
# adding EOS, SOS, PAD, OUT to dicts
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for t in tokens:
    qwords2int[t] = len(qwords2int) + 1
    awords2int[t] = len(awords2int) + 1
# create inverse mapping
int2awords = dict(zip(awords2int.values(),awords2int.keys()))

for i in range(len(answers)):
    answers[i] += ' <EOS>' 

# Convert questions and answers to int
# Replace all words that are filtered out
q2int = []
for q in questions:
    q2int.append(map(lambda w: qwords2int.get(w, qwords2int['<OUT>']), q.split()))
a2int = []
for a in answers:
    a2int.append(map(lambda w: awords2int.get(w, awords2int['<OUT>']), a.split()))

# Sorting questions and answers by length of qs
sorted_qs = []
sorted_as = []
for l in range(1, 26): # up to 25 + 1
    for i,q in enumerate(q2int):
        if len(q) == l:
            sorted_qs.append(q)
            sorted_as.append(a2int[i])

# PART 2 - SEQ2SEQ MODEL
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], 'input')
    targets = tf.placeholder(tf.int32, [None, None], 'target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob


def preprocess_targets(targets, word2int, batch_size):
    # [<SOS>] * batch size array 
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    # get batch_size number of rows with all columns except the last one
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1,1])
    return tf.concat([left_side, right_side], axis=1)

# Creating Encoder RNN Layer
def create_encoder(rnn_inputs, rnn_size, nb_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * nb_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell, 
        cell_bw=encoder_cell,
        sequence_length=sequence_length, 
        inputs=rnn_inputs,
        dtype=tf.float32) 
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state, 
                        decoder_cell, 
                        decoder_embedded_input, 
                        sequence_length, 
                        decoding_scope, 
                        output_function, 
                        keep_prob,
                        batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_function,
                                                                              attention_construct_function,
                                                                              name="attn_dec_train")
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope=decoding_scope)
    decoder_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_dropout)


# Decode the test set
def decode_test_set(encoder_state, 
                        decoder_cell, 
                        decoder_embeddings_matrix,
                        sos_id,
                        eos_id,
                        max_length,
                        nb_words,
                        decoding_scope, 
                        output_function,    
                        batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option='bahdanau', num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              max_length,
                                                                              nb_words,      
                                                                              name="attn_dec_inf")
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                  test_decoder_function,
                                                                  scope=decoding_scope)
    return test_predictions

# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, 
                decoder_embeddings_matrix, 
                encoder_state, nb_words, sequence_length, rnn_size,
                nb_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * nb_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, nb_words, None, scope=decoding_scope, weights_initializer=weights, biases_initializer=biases)
        training_pred = decode_training_set(encoder_state, decoder_cell, decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size)
        decoding_scope.reuse_variables()
        test_pred = decode_test_set(encoder_state, decoder_cell,decoder_embeddings_matrix,
                                    word2int['<SOS>'],word2int['<EOS>'],sequence_length-1,
                                    nb_words, decoding_scope, output_function, batch_size)
    return training_pred, test_pred

# Building seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_nb_words, questions_nb_words, encoder_embedding_size, decoder_embedding_size, rnn_size, nb_layers, qwords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_nb_words+1, encoder_embedding_size, initializer=tf.random_uniform_initializer(0,1))
    encoder_state = create_encoder(encoder_embedded_input, rnn_size, nb_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, qwords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_nb_words+1, decoder_embedding_size]))
    decoder_embedded_inputs = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_pred , test_pred = decoder_rnn(decoder_embedded_inputs, 
                                            decoder_embeddings_matrix, 
                                            encoder_state, 
                                            questions_nb_words, 
                                            sequence_length, 
                                            rnn_size,
                                            nb_layers, 
                                            qwords2int, 
                                            keep_prob, 
                                            batch_size)
    return training_pred, test_pred

# PART 3 - TRAINING THE SEQ2SEQ MODEL

# Setting Hyper params
epochs = 100
batch_size = 64
rnn_size = 512
nb_layers = 3
encoder_embedding_size = 512
decoding_embedding_size = 512
lr = 0.01
lr_decay = 0.9
min_lr = 0.0001
keep_prob = .5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name="sequence_length")

# Getting the shape of the input tensor
input_shape = tf.shape(inputs)

# Get training and test predictions
training_pred, test_pred = seq2seq_model(tf.reverse(inputs, [-1]), 
                                         targets,
                                         keep_prob,
                                         batch_size,
                                         sequence_length,
                                         len(awords2int),
                                         len(qwords2int),
                                         encoder_embedding_size,
                                         decoding_embedding_size,
                                         rnn_size, nb_layers, qwords2int)
# Setting up Loss, optimizer and gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_pred, targets, 
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5. , 5.), grad_var) for grad_tensor, grad_var in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding 
def apply_padding(batch, word2int):
    length = max(batch, key=lambda x:len(x)) 
    return [s + [word2int['<PAD>']]*(length-len(s)) for s in batch]

# Splitting data into batches of qs and as
def split_into_batches(qs, answers, batch_size):
    for i in range(len(qs)//batch_size):
        si = i * batch_size
        qs_in_batch = qs[si: si + batch_size]
        a_in_batch = answers[si: si + batch_size]
        qs_in_batch = np.array(apply_padding(qs_in_batch, qwords2int))
        a_in_batch = np.array(apply_padding(answers, awords2int))
        yield qs_in_batch, a_in_batch

# Split the qs and answers into training and validation sets
training_validation_split = int(len(questions) * 0.15)
training_qs = questions[training_validation_split:]
training_as = answers[training_validation_split:]
validation_qs = questions[:training_validation_split]
validation_as = answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_qs)//batch_size//2)) - 1
total_training_loss_error = 0
validation_loss_error_list = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for e in range(epochs):
    for i, (qs_in_batch, a_in_batch) in enumerate(split_into_batches(training_qs, training_as, batch_size)):
        start_time = time.time()
        _, batch_loss_error = session.run([optimizer_gradient_clipping, loss_error],{inputs:qs_in_batch,
                                                                                     targets: a_in_batch,
                                                                                     lr:lr,
                                                                                     sequence_length:a_in_batch.shape[1],
                                                                                     keep_prob:keep_prob})
        total_training_loss_error += batch_loss_error
        end = time.time()
        batch_time = end - start_time
        if i % batch_index_check_training_loss == 0:
            print("epoch: {:>3}/{}, batch:{:>4}/{}, training loss error: {:>6.3f}, training time: {:d} seconds".format(e+1, epochs, i, len(training_qs)//batch_size, total_training_loss_error/batch_index_check_training_loss, int(batch_time*batch_index_check_training_loss)))
            total_training_loss_error = 0
        if i % batch_index_check_validation_loss == 0 and i > 0:
            validation_loss_error = 0
            start_time = time.time()
            for i, (qs_in_batch, a_in_batch) in enumerate(split_into_batches(validation_qs, validation_as, batch_size)):
                batch_loss_error = session.run(loss_error,{inputs:qs_in_batch,
                                                           targets: a_in_batch,
                                                           lr:lr,
                                                           sequence_length:a_in_batch.shape[1],
                                                           keep_prob: 1})
                validation_loss_error += batch_loss_error
            end = time.time()
            batch_time = end - start_time 
            avg_validation_loss_error = validation_loss_error / (len(validation_qs) / batch_size)
            print("validation loss error: {:>6.3f}, batch validation time: {:d} seconds".format(avg_validation_loss_error, int(batch_time)))
            lr *= lr_decay
            lr = max(lr, min_lr)
            validation_loss_error_list.append(avg_validation_loss_error)
            if avg_validation_loss_error < min(validation_loss_error_list):
                print('model improved')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        break

## PART 4 - TESTING SEQ2SEQ MODEL

# Loading the weights
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Convert input to econding ints
def convert_string2int(q, word2int):
    q = clean_text(q)
    return map(lambda x: word2int.get(x, word2int['<OUT>']), q.split())

# setup the chat:
while(True):
    q = input("You: ")
    if q == "Goodbye":
        break
    q = convert_string2int(q, qwords2int)
    q = q + [qwords2int["<PAD>"]] * (20-len(q))
    fake_batch = np.zeros((batch_size, 20))
    fake_batch[0] = q
    p_a = session.run(test_pred, {inputs: fake_batch, keep_prob:keep_prob})[0]
    answer = ''
    for i in np.argmax(p_a, 1):
        if awords2int[i] == 'i':
            token = 'I'
        elif awords2int[i] == '<EOS>':
            token = '.'
        elif awords2int[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + awords2int[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)

