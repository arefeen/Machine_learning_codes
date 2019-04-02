from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import load_model 
import numpy as np
import io

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '../../data/language_interpreter/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with io.open(data_path, 'r', encoding='utf-8') as f:
	lines = f.read().split('\n')
    	for line in lines[: min(num_samples, len(lines) - 1)]:
        	input_text, target_text = line.split('\t')
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
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
	for t, char in enumerate(input_text):
		encoder_input_data[i, t, input_token_index[char]] = 1.
	for t, char in enumerate(target_text):
		decoder_input_data[i, t, target_token_index[char]] = 1.
		if t > 0:
			decoder_target_data[i, t - 1, target_token_index[char]] = 1.


encoder_model = load_model('s2s_encoder.h5')
decoder_model = load_model('s2s_decoder.h5')
model = load_model('s2s.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


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
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length
		# or find stop character.
		if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.

		# Update states
		states_value = [h, c]

	return decoded_sentence

test_input = []
with io.open(data_path, 'r', encoding='utf-8') as f:
	lines = f.read().split('\n')
    	for line in lines[(num_samples+1): (num_samples+21)]:
        	input_text, _ = line.split('\t')
		test_input.append(input_text)

encoder_test_input_data = np.zeros((len(test_input), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

for i, input_text in enumerate(test_input):
	for t, char in enumerate(input_text):
		encoder_test_input_data[i, t, input_token_index[char]] = 1.



for seq_index in range(20):
	# Take one sequence (part of the training set)
	# for trying out decoding.
	input_seq = encoder_test_input_data[seq_index: seq_index + 1]
	decoded_sentence = decode_sequence(input_seq)
	print('-')
	print('Input sentence:', test_input[seq_index])
	print('Decoded sentence:', decoded_sentence)


