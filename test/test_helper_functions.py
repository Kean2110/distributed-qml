import pytest
import numpy as np


def test_split_data_into_batches_dividable():
    # IN PROGRESS
    n_data_points = 100
    batch_size = 10
    data = np.array([(0.1*i, 0.2*i) for i in range(n_data_points)])
    