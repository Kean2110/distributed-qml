from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from typing import Union, Literal
import numpy as np
import ast

def send_value(channel: Union[Socket, BroadcastChannel], value: Union[float, int]) -> None:
    """
    Sends a numerical value to a socket/broadcast_channel.
    
    :param: channel: The socket/broadcast_channel which is used for sending
    :param: value: The value that should be sent
    """
    channel.send(str(value))
    
    
def send_dict(channel: Union[Socket, BroadcastChannel], values: dict) -> None:
    """
    Sends a dict to a socket/broadcast_channel.
    
    :param: channel: The socket/broadcast_channel which is used for sending
    :param: values: The dict that should be sent
    """
    channel.send(str(values))
    
    
def receive_dict(channel: Union[Socket, BroadcastChannel]) -> dict:
    """
    Receives a value, which is expected to be a dict.
    
    :param: channel: The socket/broadcast_channel which is used for receiving
    :returns: the dict that was received
    """ 
    str_value = channel.recv(block=True)
    return ast.literal_eval(str_value)


def check_parity(qubits: [int]) -> Literal[0,1]:
    """
    Checks the parity of a list of qubits.
    Returns 0 if there is an even number of 1s (or 0),
    Returns 1 if there is an odd number of 1s.
    
    :param: qubits: The qubit results
    :returns: 0 or 1
    """
    parity = sum(qubits) % 2
    return parity


def split_data_into_batches(data: [int], batch_size: int) -> [int]:
    """
    Splits the data into batches of size batch_size.
    If the data is not dividable by batch_size, the last batch is smaller.
    
    :param: batch_size: the size of each batch
    :data: the data samples as a list
    :returns: the batches that were created
    """
    # if the number of samples is dividable by the batch size
    if len(data) % batch_size == 0:
        return np.split(data, len(data) / batch_size)
    # calculate the number of full batches (that all have batch_size elements)
    n_full_batches = len(data) // batch_size
    # take the data points that make up the full batches
    full_batch_data = data[:n_full_batches*batch_size]
    # using split and not array_split, so an exception is raised in case len(full_batch_data) % batch_size != 0
    first_batches = np.split(full_batch_data, n_full_batches)
    # calculate the last batch
    last_batch = data[n_full_batches*batch_size:]
    # append last batch to the first batches
    first_batches.append(last_batch)
    return first_batches


if __name__ == '__main__':
    data = np.arange(160)
    batch_size = 16
    print(split_data_into_batches(data, 16))
    
    
    
    