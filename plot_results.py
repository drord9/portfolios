import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    results = pd.read_pickle('../results_pamr.pkl')
    plt.plot(results, label=results.columns)
    plt.legend(loc="best")
    plt.show()