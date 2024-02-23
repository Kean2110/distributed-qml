from netqasm.sdk.external import NetQASMConnection, BroadcastChannel, Socket
from typing import List, Tuple, Union, Literal
from netqasm.sdk import Qubit, EPRSocket
from netqasm.sdk.toolbox import set_qubit_state
from netqasm.sdk.classical_communication.message import StructuredMessage
import numpy as np
import ast
import random
import torch

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
    
 
def send_tensor(channel: Union[Socket, BroadcastChannel], value: torch.Tensor) -> None:   
    """
    Converts a pytorch tensor to float and then to string and send it to a socket/broadcast_channel.
    
    :param: channel: The socket/broadcast_channel which is used for sending
    :param: value: The tensor that should be sent
    """
    number = value.item()
    channel.send(str(number))
    
    
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


def split_data_into_batches(data: List[Tuple[float, float]], labels: List[int], batch_size: int, random_seed: int = 42) -> Tuple[List[List[Tuple[float,float]]],List[int]]:
    """
    Splits the data into batches of size batch_size.
    If the data is not dividable by batch_size, the last batch is smaller.
    
    :param: batch_size: the size of each batch
    :data: the data samples as a list
    :returns: the batches that were created
    """
    # randomly shuffles the data
    combined_data = list(zip(data, labels))
    random.seed(random_seed)
    random.shuffle(combined_data)
    shuffled_data, shuffled_labels = zip(*combined_data)
    shuffled_data, shuffled_labels = np.array(shuffled_data), np.array(shuffled_labels)
    
    # if the number of samples is dividable by the batch size
    if len(data) % batch_size == 0:
        n_batches = len(data) // batch_size
        data_batches = np.split(shuffled_data, n_batches)
        label_batches = np.split(shuffled_labels, n_batches)
        return data_batches, label_batches
    # calculate the number of full batches (that all have batch_size elements)
    n_full_batches = len(data) // batch_size
    # take the data points that make up the full batches
    full_batch_data = shuffled_data[:n_full_batches*batch_size]
    full_labels = shuffled_labels[:n_full_batches*batch_size]
    # using split and not array_split, so an exception is raised in case len(full_batch_data) % batch_size != 0
    first_batches = np.split(full_batch_data, n_full_batches)
    first_labels = np.split(full_labels, n_full_batches)
    # calculate the last batch
    last_batch = shuffled_data[n_full_batches*batch_size:]
    last_labels = shuffled_labels[n_full_batches*batch_size:]
    # append last batch to the first batches
    first_batches.append(last_batch)
    first_labels.append(last_labels)
    return first_batches, first_labels


def receive_teleported_qubit(epr_socket, classical_socket, netqasm_connection):
    with netqasm_connection:
        epr = epr_socket.recv_keep()[0]
        netqasm_connection.flush()
        
        m1, m2 = classical_socket.recv_structured().payload
        print(f"`receiver` got corrections: {m1}, {m2}")
        if m2 == 1:
            print("`receiver` will perform X correction")
            epr.X()
        if m1 == 1:
            print("`receiver` will perform Z correction")
            epr.Z()
        netqasm_connection.flush()
        return epr
    
    
def teleport_qubit(epr_socket, classical_socket, netqasm_connection, feature, theta):
    with netqasm_connection:
        q = Qubit(netqasm_connection)
        set_qubit_state(q, feature, theta)
        
        epr = epr_socket.create_keep()[0]
        
        # Teleport
        q.cnot(epr)
        q.H()
        m1 = q.measure()
        m2 = epr.measure()

    # send corrections
    m1, m2 = int(m1), int(m2)
    
    classical_socket.send_structured(StructuredMessage("Corrections", (m1,m2)))
   
   
def remote_cnot_control(epr_socket: EPRSocket, classical_socket: Socket, netqasm_conn: NetQASMConnection, control_qubit: Qubit):
    with netqasm_conn:
        # create EPR pair
        epr_ctrl = epr_socket.create_keep()[0]
        netqasm_conn.flush()
        
        # CNOT between ctrl and epr
        control_qubit.cnot(epr_ctrl)
        
        # measure epr
        epr_ctrl_meas = epr_ctrl.measure()
        netqasm_conn.flush()
        
        classical_socket.send(str(epr_ctrl_meas))
        
        # wait for target's measurement outcome to undo potential entanglement
        # between his EPR half and the control qubit
        target_meas = classical_socket.recv()
        if target_meas == "1":
            control_qubit.Z()
        netqasm_conn.flush()
        
        
def remote_cnot_target(epr_socket: EPRSocket, classical_socket: Socket, netqasm_conn: NetQASMConnection, target_qubit: Qubit):
    with netqasm_conn:
        # receive EPR qubit
        epr_target = epr_socket.recv_keep()[0]
        netqasm_conn.flush()

        # receive measurement result from EPR pair from controller
        epr_meas = classical_socket.recv()

        # apply X gate if control epr qubit is 1
        if epr_meas == "1":
            epr_target.X()

        # apply CNOT between EPR Qubit and target qubit
        epr_target.cnot(target_qubit)

        # apply H gate to epr target qubit and measure it and send it to controller
        epr_target.H()

        # undo any potential entanglement between `epr` and controller's control qubit
        epr_target_meas = epr_target.measure()
        netqasm_conn.flush()

        # Controller will do a controlled-Z based on the outcome to undo the entanglement
        classical_socket.send(str(epr_target_meas))
        
        
        
    
 

if __name__ == '__main__':
    n = 100
    data = np.arange(n)
    labels = np.concatenate((np.zeros(n//2, int), np.ones(n//2, int)))
    batch_size = 16
    print(split_data_into_batches(data, labels, batch_size, 42))
    
    
    
    