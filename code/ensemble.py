import tensorflow as tf
import numpy as np
from dataset import dataset_oil


def main():
    x, y = dataset_oil()
    print(x)


if __name__ == "__main__":
    main()
