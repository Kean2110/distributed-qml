from netqasm.sdk import Qubit
from netqasm.sdk.external import NetQASMConnection
from preprocessing import preprocessing_main
import numpy as np

def encode_in_circuit(image: np.ndarray):
    flattened_image = np.ndarray.flatten(image)
    print(flattened_image)
    conn = NetQASMConnection(app_name = "encoding")
    circuit = []
    with conn:
        outcomes = conn.new_array(len(flattened_image))
        values = conn.new_array(len(flattened_image), init_values=flattened_image)
        circuit = [Qubit(conn)] * len(values)
        with values.foreach() as v:
            q = Qubit(conn=conn)
            with v.if_eq(1):
                q.X()
    return circuit
    '''
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit
    '''
    
def build_encoding():
    (x_train, y_train), (x_test, y_test) = preprocessing_main()
    circ = encode_in_circuit(x_train[0])
    print(circ)
    #for img in x_train:
    #    encode_in_circuit(img)
        
if __name__ == "__main__":
    build_encoding()