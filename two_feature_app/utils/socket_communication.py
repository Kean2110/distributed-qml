import ast
import numpy as np
import torch
from typing import Literal, Union
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import BroadcastChannel, Socket


def send_value(channel: Union[Socket, BroadcastChannel], value: Union[float, int]) -> None:
    """
    Sends a numerical value to a socket/broadcast_channel.

    :param: channel: The socket/broadcast_channel which is used for sending
    :param: value: The value that should be sent
    """
    channel.send(str(value))


def send_as_str(channel: Union[Socket, BroadcastChannel], values: Union[str, int, float, bool, object]) -> None:
    """
    Sends a value as a string to a socket/broadcast_channel.

    :param: channel: The socket/broadcast_channel which is used for sending
    :param: values: The value(s) that should be sent
    """
    channel.send(str(values))
    
    
def receive_and_eval(channel: Union[Socket, BroadcastChannel]) -> Union[bytes, int, float, list, dict, set, bool]:
    """
    Receives a value, which is evaluated.

    :param: channel: The socket/broadcast_channel which is used for receiving
    :returns: the literal that was received
    """
    str_value = channel.recv(block=True)
    return ast.literal_eval(str_value)


def send_tensor(channel: Union[Socket, BroadcastChannel], value: torch.Tensor) -> None:
    """
    Converts a pytorch tensor to float and then to string and send it to a socket/broadcast_channel.

    :param: channel: The socket/broadcast_channel which is used for sending
    :param: value: The tensor that should be sent
    """
    number = value.item()
    channel.send(str(number))


def send_with_header(channel: Union[Socket, BroadcastChannel], payload: Union[dict, int, list, np.ndarray], header: str) -> None:
    if isinstance(payload, np.ndarray):
        # send as list
        payload_str = str(payload.tolist())
    else:
        payload_str = str(payload)
    msg = StructuredMessage(header=header, payload=payload_str)
    channel.send_structured(msg)


def receive_with_header(channel: Union[Socket, BroadcastChannel], expected_header: str, expected_dtype: Union[dict, int, list, np.ndarray] = None) -> tuple[str,Union[dict, int, list]]:
    msg = channel.recv_structured(block=True)
    assert msg.header == expected_header
    payload_evaluated = ast.literal_eval(msg.payload)
    # since we only send lists as strings, we have to transform it back to numpy array
    if expected_dtype == np.ndarray:
        payload_evaluated = np.array(payload_evaluated)
    return payload_evaluated


