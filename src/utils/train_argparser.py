import argparse

def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", help="Number of epochs for train run")
    parser.add_argument("--batch_size", help="Batch size for images")
    parser.add_argument("--lr", help="Learning rate for the optimiser")

    return parser