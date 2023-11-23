from netqasm.sdk import Qubit
from preprocessing import preprocessing_main
import numpy as np

def encode_in_circuit(image: np.ndarray):
    values = np.ndarray.flatten(image)
    print(values)
    circuit = [Qubit] * len(values)
    print(circuit)
    for i, qbit in enumerate(circuit):
        if values[i]:
            qbit.X()
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
    #for img in x_train:
    #    encode_in_circuit(img)
        
if __name__ == "__main__":
    build_encoding()