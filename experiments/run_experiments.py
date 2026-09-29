"""Run the full laptop-sized, predeclared experiment matrix."""

from tabular_benchmark.experiments import run_all


if __name__ == "__main__":
    for artifact, path in run_all().items():
        print(f"{artifact}: {path}")
