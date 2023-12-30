from netqasm.sdk.external import NetQASMConnection
from netqasm.sdk import Qubit


def main(app_config=None):
   # Setup a connection to QNodeOS
    with NetQASMConnection("alice") as alice:
        # Create a qubit
        q = Qubit(alice)
        # Perform a Hadamard gate
        q.H()
        # Measure the qubit
        m = q.measure()
        # Flush the current subroutine
        alice.flush()
        # Print the outcome
        print(f"Outcome is: {m}")