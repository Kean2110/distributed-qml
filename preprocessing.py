import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import collections
import config


def load_mnist_data() -> tuple[np.ndarray, np.ndarray]:
    return tf.keras.datasets.mnist.load_data()

def rescale_images(data_array: np.ndarray, factor: float) -> np.ndarray:
    rescaled_array = data_array[..., np.newaxis]/factor
    return rescaled_array

def filter_zeros_and_ones(data_array: np.ndarray, labels_array: np.ndarray) -> (np.ndarray, np.ndarray):
    if len(data_array) != len(labels_array):
        raise ValueError('data_array and labels_array must have the same length.')
    filter_mask = np.isin(labels_array, [0,1])
    data_filtered = data_array[filter_mask]
    labels_filtered = labels_array[filter_mask]
    return (data_filtered, labels_filtered)
    
def downfilter_images(data_array: np.array, size: (int, int)) -> np.array:
    x_train_small = tf.image.resize(data_array, size).numpy()
    return x_train_small

def remove_contradicting_samples(xs: np.ndarray, ys: np.ndarray) -> (np.ndarray, np.ndarray):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)

    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass

    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print()
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))

    return np.array(new_x), np.array(new_y)

def encode_array(data: np.ndarray, threshold: int = config.ENCODING_THRESHOLD) -> np.ndarray:
    data_bin = np.array(data > threshold, dtype=np.float32)
    return data_bin

def preprocessing_binary_encoding() -> tuple[np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
    (x_train, y_train) = filter_zeros_and_ones(x_train, y_train)
    (x_test, y_test) = filter_zeros_and_ones(x_test, y_test)
    x_train = rescale_images(x_train, 255.0)
    x_test = rescale_images(x_test, 255.0)
    print("Number of training samples ", str(len(y_train)))
    print("Number of testing samples: ", str(len(y_test)))
    x_train = downfilter_images(x_train, (4,4))
    x_test = downfilter_images(x_test, (4,4))
    if config.REMOVE_CONTRADICTING:
        x_train, y_train = remove_contradicting_samples(x_train, y_train)
    x_train_bin = encode_array(x_train)
    x_test_bin = encode_array(x_test)
    return (x_train_bin, y_train), (x_test_bin, y_test)

def preprocessing_amplitude_encoding() -> tuple[np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
    (x_train, y_train) = filter_zeros_and_ones(x_train, y_train)
    (x_test, y_test) = filter_zeros_and_ones(x_test, y_test)
    x_train = rescale_images(x_train, 255.0)
    x_test = rescale_images(x_test, 255.0)
    print("Number of training samples ", str(len(y_train)))
    print("Number of testing samples: ", str(len(y_test)))
    x_train = downfilter_images(x_train, (4,4))
    x_test = downfilter_images(x_test, (4,4))
    if config.REMOVE_CONTRADICTING:
        x_train, y_train = remove_contradicting_samples(x_train, y_train)
    return (x_train, y_train), (x_test, y_test)
    

if __name__ == "__main__":  
    start = time.process_time()
    preprocessing_amplitude_encoding()
    end = time.process_time()
    elapsed = end - start
    print("Processing time: ", elapsed)
    