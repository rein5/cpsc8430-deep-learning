#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = []
    for file in ['inception_score_dcgan.csv', 'inception_score_wgan.csv', 'inception_score_acgan.csv']:
        data.append(pd.read_csv(file))

    plt.plot(data[0]['epoch'], data[0][' inception_score '], label="DCGAN")
    plt.plot(data[1]['epoch'], data[1][' inception_score '], label="WGAN")
    plt.plot(data[2]['epoch'], data[2][' inception_score '], label="ACGAN")

    plt.xlabel("Epoch")
    plt.ylabel("Inception Score")
    plt.legend()
    plt.savefig("training_scores.png")
    print("Image training_scores.png exported!")

if __name__ == "__main__":
    main()
