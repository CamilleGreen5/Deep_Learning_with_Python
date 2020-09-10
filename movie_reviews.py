import tensorflow
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
# print('Name GPU available: ', tensorflow.config.experimental.list_physical_devices('GPU'))
# tensorflow.debugging.set_log_device_placement(True)

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000, )

word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

coded_review = train_data[0]
decoded_review = ''.join(
    reverse_word_index.get(i - 3, '?') + ' ' for i in coded_review)


# print(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# x_train = x_train[:1000]
# y_train = y_train[:1000]

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print('len x_train = ', len(partial_x_train))
print('len x_val = ', len(x_val))

# for e in range(10):
with tensorflow.device('/GPU:0'):
    history_train = model.fit(partial_x_train, partial_y_train, epochs=1, batch_size=10)
    # history_val = model.evaluate(x_val, y_val, batch_size=10)


# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
# epochs = range(1, len('acc') + 1)
# plt.plot(epochs, loss_values, 'bo', label='Training_loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation_loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
