import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys  #for commandline args
# data from loc 2
def plot(path):

    """Import data"""
    df = pd.read_csv(path, delimiter=",")
    df.columns = ["original_index", "rss", "epc", "time", "diff"]
    df["time"].astype('int')

    t1 = df[df["epc"] == 9107]
    t2 = df[df["epc"] == 9108]
    t1 = t1.reset_index()
    t2 = t2.reset_index()

    startTime = t1["time"][0] if t1["time"][0] < t2["time"][0] else t2["time"][0]
    t1["time"] = t1["time"].apply(lambda x: x - startTime)
    t2["time"] = t2["time"].apply(lambda x: x - startTime)

    t1 = t1.drop(columns=["index", "epc"])
    t2 = t2.drop(columns=["index", "epc"])

    """plot"""


    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(t1["rss"], label='Tag 1')
    ax1.plot(t2["rss"], linestyle='dashed', label='Tag 2')
    ax1.set_yticks(np.arange(-65, -47, 2)) 
    ax1.legend()

    ax2.plot(t1["diff"], label='Tag 1')
    ax2.plot(t2["diff"], linestyle='dashed', label='Tag 2')
    ax2.set_yticks(np.arange(-10, 6, 2)) 
    ax2.legend()

    ax1.set_ylabel("RSS (dBm)")
    ax2.set_ylabel("Derivative of RSS (dB)")
    plt.show()

if len(sys.argv) <= 1:
    print("must provide data file path")
else:
    plot(sys.argv[1])
