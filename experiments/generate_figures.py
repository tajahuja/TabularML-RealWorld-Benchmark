"""Build figures from the generated results tables."""

from tabular_benchmark.plots import generate_figures


if __name__ == "__main__":
    for path in generate_figures():
        print(path)
