#phyphox configuration
PP_ADDRESS = "http://10.216.245.5"
PP_CHANNELS = ["accX", "accY", "accZ"]

import requests
import time
import pandas as pd
import numpy as np


column_names = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                    "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]

df = pd.DataFrame(columns=column_names)

startTime = time.time()

while True:
    url = PP_ADDRESS + "/get?" + ("&".join(PP_CHANNELS))
    data = requests.get(url=url).json()
    currentTime = time.time()
    accX = data["buffer"][PP_CHANNELS[0]]["buffer"][0]
    accY = data["buffer"][PP_CHANNELS[1]]["buffer"][0]
    accZ = data["buffer"][PP_CHANNELS[2]]["buffer"][0]
    absAcc = np.sqrt(accX**2 + accY**2 + accZ**2)
    df.loc[len(df.index)] = [currentTime, accX, accY, accZ, absAcc]
    print([time.time() - startTime, accX, accY, accZ, absAcc])
    if currentTime - startTime >= 5:
        startTime = time.time()
        print(df)
        df = pd.DataFrame(columns=column_names)
