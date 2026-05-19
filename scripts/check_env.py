from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"python version: {sys.version.split()[0]}")
    print(f"python executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")

    try:
        import torch
    except Exception as exc:
        print(f"torch import error: {exc}")
        return

    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        print(f"gpu name: {props.name}")
        print(f"gpu memory: {props.total_memory / (1024 ** 3):.2f} GiB")
    else:
        print("gpu name: none")
        print("gpu memory: 0 GiB")

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    print(f"mps available: {mps_available}")

    try:
        import torch_geometric

        print(f"torch_geometric version: {torch_geometric.__version__}")
    except Exception as exc:
        print(f"torch_geometric import error: {exc}")


if __name__ == "__main__":
    main()
