"""Command-line entry point for quick environment diagnostics."""
from rcdpo.device import DEVICE, DTYPE
from rcdpo.paths import PROJECT_ROOT


def main() -> None:
    print(f"project_root={PROJECT_ROOT}")
    print(f"device={DEVICE} dtype={DTYPE}")


if __name__ == "__main__":
    main()
