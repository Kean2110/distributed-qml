from netqasm.runtime.application import default_app_instance
from netqasm.runtime.app_config import default_app_config
from netqasm.runtime.hardware import get_netqasm_logger
from netqasm.sdk.external import simulate_application
from netqasm.sdk.config import LogConfig
from netqasm.runtime.debug import run_application
import app_node1
import app_node2


def create_app():
    app_instance = default_app_instance(
        [("node1", app_node1.main), ("node2", app_node2.main)]
    )

    log_config = LogConfig(log_dir="log/")

    app_instance.logging_cfg = log_config

    simulate_application(
        app_instance, use_app_config=True, post_function=None, enable_logging=False
    )


if __name__ == "__main__":
    create_app()
