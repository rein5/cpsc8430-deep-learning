#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = []
    for file in ['inception_score_dcgan.csv', 'inception_score_wgan.csv', 'inception_score_acgan.csv']:
        data.append(pd.read_csv(file))

    plt.plot(data[0]['Iteration'], data[0]['Inception Score'], label="DCGAN")
    plt.plot(data[1]['Iteration'], data[1]['Inception Score'], label="WGAN")
    plt.plot(data[2]['Iteration'], data[2]['Inception Score'], label="ACGAN")

    plt.xlabel("Epoch")
    plt.ylabel("Inception Score")
    plt.legend()
    plt.savefig("training_scores.png")

if __name__ == "__main__":
    main()
