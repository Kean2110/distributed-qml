import math
from netqasm.sdk import Qubit
from netqasm.sdk.external import Socket
from utils.logger import logger
from utils.helper_functions import phase_gate
from utils.qubit_communication import remote_cnot_control, remote_cnot_target

def zz_feature_map_ctrl(self, qubit: Qubit, eprs: list[Qubit], feature: float, socket_other_client: Socket):
    qubit.H()
    phase_gate(2 * feature, qubit)
    remote_cnot_control(socket_other_client, qubit.connection, qubit, eprs[0])
    remote_cnot_control(socket_other_client, qubit.connection, qubit, eprs[1])
    logger.debug(f"{qubit.connection.app_name} executed zz feature map control")
        
    
def zz_feature_map_target(self, qubit: Qubit, eprs: list[Qubit], feature: float, feature_other_node: list[float], socket_other_client: Socket):
    qubit.H()
    phase_gate(angle = 2 * feature, qubit = qubit)
    remote_cnot_target(socket_other_client, qubit.connection, qubit, eprs[0])
    phase_gate(angle = 2 * (math.pi - feature_other_node) * (math.pi - feature), qubit = qubit)
    remote_cnot_target(socket_other_client, qubit.connection, qubit, eprs[0])
    logger.debug(f"{qubit.connection.app_name} executed zz feature map target")
        
        
def ry_feature_map(qubit: Qubit, feature: float):
    qubit.rot_Y(angle=feature)