import math
import threading
import requests
import time
import sys
import os
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,f1_score,roc_curve,roc_auc_score,RocCurveDisplay
from sklearn.linear_model import LogisticRegression

import matplotlib
import PyQt5
from PyQt5 import QtWidgets, QtCore #pyqt stuff
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QToolBar, QAction, QStatusBar, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, QGridLayout, QFileDialog,
    QLineEdit
)
from PyQt5.QtCore import Qt, QSize

matplotlib.use('Qt5Agg')
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

if os.path.exists("hdf5_data.h5"):
    os.remove("hdf5_data.h5")
def segment_data(df, window_size=5):
    # Calculate the number of windows
    num_windows = int(df['Time (s)'].iloc[-1] // window_size) + 1

    # Initialize an empty list to hold the dataframes
    grouped = []

    # Loop over each window
    for i in range(num_windows):
        # Get the start and end times for this window
        start_time = i * window_size
        end_time = (i + 1) * window_size

        # Create a new dataframe for this window
        window_df = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] < end_time)].copy()
        # Append the new dataframe to the list
        grouped.append(window_df)

    return grouped

# Function to shuffle the segmented data
def shuffle_data(data):
    np.random.shuffle(data)
    return data


def visualization(df_walk, df_jump):
    column_names = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                    "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)", "Bool"]
    # Plotting walking data
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df_walk[column_names[0]], df_walk[column_names[1]],
                 label="Linear Acceleration x (m/s^2)")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df_walk[column_names[0]], df_walk[column_names[2]],
                 label="Linear Acceleration y (m/s^2)", color='orange')
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(df_walk[column_names[0]], df_walk[column_names[3]],
                 label="Linear Acceleration z (m/s^2)", color='green')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].legend()
    axes[2].grid(True)
    plt.suptitle(f'Accelerometer Data Over Time for Combined Group Members Walking', fontsize=20)

    # Plotting jumping data
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df_jump[column_names[0]], df_jump[column_names[1]],
                 label="Linear Acceleration x (m/s^2)")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df_jump[column_names[0]], df_jump[column_names[2]],
                 label="Linear Acceleration y (m/s^2)", color='orange')
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(df_jump[column_names[0]], df_jump[column_names[3]],
                 label="Linear Acceleration z (m/s^2)", color='green')
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m/s^2)")
    axes[2].legend()
    axes[2].grid(True)
    plt.suptitle(f'Accelerometer Data Over Time for Combined Group Members Jumping', fontsize=20)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(df_walk[column_names[0]], df_walk[column_names[4]],
                 label="Absolute Acceleration for Walking (m/s^2)")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(df_jump[column_names[0]], df_jump[column_names[4]],
                 label="Absolute Acceleration for Jumping (m/s^2)", color='orange')
    axes[1].set_ylabel("Acceleration (m/s^2)")
    axes[1].legend()
    axes[1].grid(True)
    plt.suptitle(f'Absolute Acceleration Data Over Time for Combined Group Members', fontsize=20)
    #plt.show()

#visualization(big_df_walking, big_df_jumping)

def extract(df_array):
    features = []

    for window in df_array:
        features.append({
            'mean_abs': window["Absolute acceleration (m/s^2)"].mean(),
            'mean_x': window["Linear Acceleration x (m/s^2)"].mean(),
            'mean_y': window["Linear Acceleration y (m/s^2)"].mean(),
            'mean_xy': np.sqrt(window["Linear Acceleration x (m/s^2)"].mean()**2 + window["Linear Acceleration y (m/s^2)"].mean()**2),
            'mean_xz': np.sqrt(window["Linear Acceleration x (m/s^2)"].mean()**2 + window["Linear Acceleration z (m/s^2)"].mean()**2),
            'mean_yz': np.sqrt(window["Linear Acceleration y (m/s^2)"].mean()**2 + window["Linear Acceleration z (m/s^2)"].mean()**2),
            'angle_xy': np.tan(window["Linear Acceleration y (m/s^2)"].mean()/window["Linear Acceleration x (m/s^2)"].mean()),
            'angle_xz': np.tan(window["Linear Acceleration z (m/s^2)"].mean() / window["Linear Acceleration x (m/s^2)"].mean()),
            'angle_yz': np.tan(window["Linear Acceleration z (m/s^2)"].mean() / window["Linear Acceleration y (m/s^2)"].mean()),
            'median_y': window["Absolute acceleration (m/s^2)"].median(),
            'std_x': window["Linear Acceleration x (m/s^2)"].std(),
            'std_y': window["Linear Acceleration y (m/s^2)"].std(),
            'std_z': window["Linear Acceleration z (m/s^2)"].std(),
            'max_y': window["Linear Acceleration y (m/s^2)"].max(),
            'min_y': window["Linear Acceleration y (m/s^2)"].min(),
            'kurtosis_y': window["Linear Acceleration y (m/s^2)"].kurtosis(),
            'skew_y': window["Linear Acceleration y (m/s^2)"].skew(),
            'range_x': (window["Linear Acceleration x (m/s^2)"].max() - window["Linear Acceleration x (m/s^2)"].min()),
            'range_z': (window["Linear Acceleration z (m/s^2)"].max() - window["Linear Acceleration z (m/s^2)"].min()),
            'variance_y': window["Linear Acceleration y (m/s^2)"].var()
        })
    df_features = pd.DataFrame(features)
    df_features.dropna(inplace=True)
    return df_features


def boolMode(df_array):
    list_df = []
    for window in df_array:
        list_df.append({
            'Bool': window.mode()[0]
        })
    dF = pd.DataFrame(list_df)
    dF.dropna(inplace=True)
    return dF

# File paths of CSV files
csv_files = [
    "member1_walking.csv",
    "member1_jumping.csv",
    "member2_walking.csv",
    "member2_jumping.csv",
    "member3_walking.csv",
    "member3_jumping.csv"
]

# Put CSV file data into panda DataFrame
dF_walking1 = pd.read_csv("member1_walking.csv")
dF_walking1["Bool"] = 0
dF_jumping1 = pd.read_csv("member1_jumping.csv")
dF_jumping1["Bool"] = 1
dF_walking2 = pd.read_csv("member2_walking.csv")
dF_walking2["Bool"] = 0
dF_jumping2 = pd.read_csv("member2_jumping.csv")
dF_jumping2["Bool"] = 1
dF_walking3 = pd.read_csv("member3_walking.csv")
dF_walking3["Bool"] = 0
dF_jumping3 = pd.read_csv("member3_jumping.csv")
dF_jumping3["Bool"] = 1

# Put all data frames in a single array for ease of removing data
all_dFs = [dF_walking1, dF_jumping1, dF_walking2, dF_jumping2, dF_walking3, dF_jumping3]
all_dFs_walking = [dF_walking1, dF_walking2, dF_walking3]
all_dFs_jumping = [dF_jumping1, dF_jumping2, dF_jumping3]

# Then into one big dataframe for a big dataset:
big_df = pd.concat(all_dFs, ignore_index=True)
big_df_walking = pd.concat(all_dFs_walking, ignore_index=True)
big_df_jumping = pd.concat(all_dFs_jumping, ignore_index=True)

# Verify dataset has no NaN to ensure proper measurement has been conducted
print("Check datset for non number values in the columns: ")
print(big_df.isna().sum())
print()

# Shuffle and segmented data into 5 second intervals:
# segmented_data = segment_data(big_df) # when you concatenate everything there will be conflicts
segmented_data_walking = segment_data(big_df_walking) # we should segment walking and jumping data separately before concatenating
segmented_data_jumping = segment_data(big_df_jumping)
segmented_data = segmented_data_walking
segmented_data.extend(segmented_data_jumping)

#shuffled_data = shuffle_data(segmented_data)
#final_data = pd.concat(shuffled_data)
final_data = shuffle_data(segmented_data)
# you can combine arrays of walking and jumping intervals instead
print(final_data)
#train_data, test_data = train_test_split(final_data, test_size=0.1, random_state=0)

#instead of train_test_split,
train_data = final_data[:math.floor(len(final_data)*0.90)] #train data would be an array of data in segmented 5-second intervals
test_data = final_data[math.floor(len(final_data)*0.90):] #test data would be an array of data in segmented 5-second intervals

# Write to H5PY File
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    # Create a group for original dataset of member1
    G1 = hdf.create_group('/member1')
    G1.create_dataset('walkingDs1', data=dF_walking1)
    G1.create_dataset('jumpingDs1', data=dF_jumping1)

    # Create a group for original dataset of member2
    G2 = hdf.create_group('/member2')
    G2.create_dataset('walkingDs2', data=dF_walking2)
    G2.create_dataset('jumpingDs2', data=dF_jumping2)

    # Create a group for original dataset of member3
    G3 = hdf.create_group('/member3')
    G3.create_dataset('walkingDs3', data=dF_walking3)
    G3.create_dataset('jumpingDs3', data=dF_jumping3)

    train_group = hdf.create_group('/dataset/training')

    # Combine all the windowed data into a single dataset for training
    #train_data_combined = pd.concat(train_data)

    smallest_num_rows = sys.maxsize
    train_data = [window for window in train_data if len(window.index) >= 1000]
    for i in range(len(train_data)):
        if len(train_data[i].index) < smallest_num_rows:
            smallest_num_rows = len(train_data[i].index)
    print(f"Smallest number of rows is {smallest_num_rows}")
    for i in range(len(train_data)):
        train_data[i] = train_data[i].iloc[:smallest_num_rows, :]

    train_group.create_dataset('training_data', data=train_data)
    #we're gonna have to truncate some of the points such that it has a consistent dimension

    # Create a group for the testing dataset

    test_group = hdf.create_group('/dataset/testing')
    smallest_num_rows = sys.maxsize
    test_data = [window for window in test_data if len(window.index) >= 1000]
    for i in range(len(test_data)):
        if len(test_data[i].index) < smallest_num_rows:
            smallest_num_rows = len(test_data[i].index)
    print(f"Smallest number of rows is {smallest_num_rows}")
    for i in range(len(test_data)):
        test_data[i] = test_data[i].iloc[:smallest_num_rows, :]


    # Combine all the windowed data into a single dataset for testing
    #test_data_combined = pd.concat(test_data)
    test_group.create_dataset('test_data', data=test_data)

"""
for i in range(3):
    with h5py.File('./hdf5_data.h5', 'r') as hdf:
        items = list(hdf.items())
        print("HDF5 File Structure:")
        print(items)
        print()

        membNum = i+1
        memb = f'member{membNum}'

        print(f'{memb}')
        G1 = hdf.get(f'/{memb}')
        print(list(G1.items()))
        print()

        # Walking data for member 1
        memWalking = hdf[f'/{memb}/walkingDs{membNum}']
        column_names = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                        "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)", "Bool"]
        walking_data = pd.DataFrame(memWalking, columns=column_names)
        walking_data.drop("Absolute acceleration (m/s^2)", axis=1, inplace=True)

        # Jumping data
        memjumping = hdf[f'/{memb}/jumpingDs{membNum}']
        jumping_data = pd.DataFrame(memjumping, columns=column_names)
        jumping_data.drop("Absolute acceleration (m/s^2)", axis=1, inplace=True)

        # Plotting walking data
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(walking_data["Time (s)"], walking_data["Linear Acceleration x (m/s^2)"],
                     label="Linear Acceleration x (m/s^2)")
        axes[0].set_ylabel("Acceleration (m/s^2)")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(walking_data["Time (s)"], walking_data["Linear Acceleration y (m/s^2)"],
                     label="Linear Acceleration y (m/s^2)", color='orange')
        axes[1].set_ylabel("Acceleration (m/s^2)")
        axes[1].legend()
        axes[1].grid(True)
        axes[2].plot(walking_data["Time (s)"], walking_data["Linear Acceleration z (m/s^2)"],
                     label="Linear Acceleration z (m/s^2)", color='green')
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Acceleration (m/s^2)")
        axes[2].legend()
        axes[2].grid(True)
        plt.suptitle(f'Accelerometer Data Over Time for Group Member {membNum} Walking', fontsize=20)

        # Plotting jumping data
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(jumping_data["Time (s)"], jumping_data["Linear Acceleration x (m/s^2)"],
                     label="Linear Acceleration x (m/s^2)")
        axes[0].set_ylabel("Acceleration (m/s^2)")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].plot(jumping_data["Time (s)"], jumping_data["Linear Acceleration y (m/s^2)"],
                     label="Linear Acceleration y (m/s^2)", color='orange')
        axes[1].set_ylabel("Acceleration (m/s^2)")
        axes[1].legend()
        axes[1].grid(True)
        axes[2].plot(jumping_data["Time (s)"], jumping_data["Linear Acceleration z (m/s^2)"],
                     label="Linear Acceleration z (m/s^2)", color='green')
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Acceleration (m/s^2)")
        axes[2].legend()
        axes[2].grid(True)
        plt.suptitle(f'Accelerometer Data Over Time for Group Member {membNum} Jumping', fontsize=20)
        #plt.show()
"""

with h5py.File('./hdf5_data.h5', 'r') as hdf:
    print("'dataset' Group Structure:")
    # reading in datasets members:
    dataSet_group = hdf.get('/dataset')
    print(list(dataSet_group.items()))
    print()

    # Training dataset read:
    print("Training Data Structure must have (x rows,5 col):")
    dTraining = dataSet_group.get('training')
    dTraining = [dataset.shape for name1, dataset in dTraining.items()]
    print(dTraining)
    print()

    # Testing dataset read:
    print("Testing Data Structure must have (x rows,5 col):")
    dTesting = dataSet_group.get('testing')
    dTesting = [dataset.shape for name2, dataset in dTesting.items()]
    print(dTesting)
    print()

with h5py.File('./hdf5_data.h5', 'r') as hdf:
    memtraining = hdf['/dataset/training/training_data']
    memtesting = hdf['/dataset/testing/test_data']

    column_names = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                    "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)", "Bool"]

    dF_training = [pd.DataFrame(window, columns=column_names) for window in memtraining]
    for df in dF_training:
        df.drop("Time (s)",axis=1,inplace=True)

    #dF_training = pd.DataFrame(memtraining, columns=column_names)
    print(dF_training)
    dF_testing = [pd.DataFrame(window, columns=column_names) for window in memtesting]
    for df in dF_testing:
        df.drop("Time (s)",axis=1,inplace=True)
    print(dF_testing)
    print("df testing done")
    #dF_testing = pd.DataFrame(memtesting, columns=column_names)
    # Do not combine this shit oh my god
    # do normalization and denoising functions on testing and training data separately



    # do rolling average in every single interval

    # Rolling mean with window 5
    window_size = 5
    train_sma = [segment.rolling(window_size).mean().dropna() for segment in dF_training]
    test_sma = [segment.rolling(window_size).mean().dropna() for segment in dF_testing]
    y_train_sma = [segment['Bool'] for segment in train_sma]
    y_test_sma = [segment['Bool'] for segment in test_sma]
    x_train_sma = [segment.drop(columns=['Bool']) for segment in train_sma]
    x_test_sma = [segment.drop(columns=['Bool']) for segment in test_sma]
    # apply rolling average to every column except the label

    # Initialize the features DataFrame

    # Rolling window size

    train_features_df = extract(x_train_sma)
    test_features_df = extract(x_test_sma)

    print(train_features_df)
    print(test_features_df)

    sc = preprocessing.StandardScaler()

    trainFit = sc.fit(train_features_df)

    trainNorm = trainFit.transform(train_features_df)
    testNorm = trainFit.transform(test_features_df)

    print(trainNorm)
    print(testNorm)

    y_train_df = boolMode(y_train_sma)
    y_test_df = boolMode(y_test_sma)
    y_train = y_train_df.values.flatten()
    y_test = y_test_df.values.flatten()

    l_reg = LogisticRegression(max_iter=10000)
    l_reg.fit(trainNorm, y_train_df.values.flatten())
    y_pred = l_reg.predict(testNorm)
    y_clf_prob = l_reg.predict_proba(testNorm)
    print(y_pred)
    print(y_test)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    print("Accuracy Score: %.2f" % accuracy_score(y_test, y_pred))
    print("F1 score: %.2f" % f1_score(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=l_reg.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    auc = roc_auc_score(y_test, y_clf_prob[:, 1])
    print("auc score: %.2f" % auc)
    #plt.show()

class MainWindow(QMainWindow):

    def upload(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getOpenFileName(self, "Select file", "", "CSV (*.csv)",
                                                  options=options)
        if filename:
            print("file found")
            self.df = pd.read_csv(filename)
            self.plot_input()
            self.segmented_data_input = segment_data(self.df)
            for window in self.segmented_data_input:
                window.drop("Time (s)", axis=1, inplace=True)
            input_sma = [segment.rolling(5).mean().dropna() for segment in self.segmented_data_input]
            input_features_df = extract(input_sma)

            in_norm = trainFit.transform(input_features_df)
            print(in_norm)
            self.input_predict = l_reg.predict(in_norm)
            self.plot_output()

    def download(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filename, _ = QFileDialog.getSaveFileName(self, "Save file", r"H:\Documents", "CSV (.csv);;All Files ()",
                                                  options=options)
        if filename:
            print(filename)
            print("it works")
            print(self.input_predict)
            output = pd.DataFrame({'Time interval (s):': [f"{5*n} - {5*n+5}" for n in range(len(self.input_predict))], "Classifier Output": ["Walking" if n == 0 else "Jumping" for n in self.input_predict]})
            output.to_csv(path_or_buf=filename)

    def plot_output(self):
        print("we plottin the output")
        self.output_figure.clear()

        ax = self.output_figure.add_subplot(111)
        val = np.array([int(x) for x in self.input_predict])
        t = np.array([5*n for n in range(len(self.input_predict))])
        x = np.arange(0, t[-1]+1)
        y = np.array([1 if val[int(round(i/5))] == 1 else 0 for i in x])
        ax.plot(x, y)
        #self.output_figure.suptitle("Walking or Jumping Classifier Output", fontsize=12)
        print("output done")
        # refresh canvas
        self.output_canvas.draw()

    def plot_input(self):
        print("we plottin")
        # clearing old figure
        self.input_figure.clear()

        x_axes = self.input_figure.add_subplot(311)

        x_axes.plot(self.df["Time (s)"], self.df["Linear Acceleration x (m/s^2)"],
                    label="Linear Acceleration x (m/s^2)")
        x_axes.legend(loc=2, prop={'size': 6})
        x_axes.grid(True)

        y_axes = self.input_figure.add_subplot(312)

        y_axes.plot(self.df["Time (s)"], self.df["Linear Acceleration y (m/s^2)"],
                    label="Linear Acceleration y (m/s^2)", color='orange')
        y_axes.set_ylabel("Acceleration (m/s^2)")
        y_axes.legend(loc=2, prop={'size': 6})
        y_axes.grid(True)

        z_axes = self.input_figure.add_subplot(313)

        z_axes.plot(self.df["Time (s)"], self.df["Linear Acceleration z (m/s^2)"],
                    label="Linear Acceleration z (m/s^2)", color='green')
        z_axes.set_xlabel("Time (s)")
        z_axes.legend(loc=2, prop={'size': 6})
        z_axes.grid(True)

        self.input_figure.suptitle("Accelerometer Data Over Time for Input Data", fontsize=12)
        self.input_figure.tight_layout()
        print("ok")
        # refresh canvas
        self.input_canvas.draw()


    def toggle_bonus(self, checked):
        if self.bonus_window.isVisible():
            self.bonus_window.hide()

        else:
            self.bonus_window.show()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.segmented_data_input = None
        self.df = None
        self.input_predict = None
        self.bonus_window = BonusWindow()

        input_plot = QWidget()

        self.input_figure = plt.figure()
        self.input_canvas = FigureCanvas(self.input_figure)
        input_toolbar = NavigationToolbar(self.input_canvas, input_plot)

        input_plot_layout = QVBoxLayout()
        input_plot_layout.addWidget(input_toolbar)
        input_plot_layout.addWidget(self.input_canvas)
        input_plot_layout.setAlignment(Qt.AlignHCenter)
        input_plot.setLayout(input_plot_layout)

        self.setWindowTitle("Jumping/Walking Classifier")

        layout_parent = QHBoxLayout()

        button_layout = QHBoxLayout()
        upload = QPushButton("Upload Data")
        download = QPushButton("Download Output")
        toggle_bonus = QPushButton("Connect Live Data")
        toggle_bonus.clicked.connect(self.toggle_bonus)
        button_layout.addWidget(upload)
        button_layout.addWidget(download)
        button_layout.addWidget(toggle_bonus)
        button_layout.setAlignment(Qt.AlignHCenter)

        button_widget = QWidget()
        button_widget.setLayout(button_layout)

        input_text = QLabel("Input Data Plot")
        input_text.setAlignment(Qt.AlignHCenter)
        in_font = input_text.font()
        in_font.setPointSize(12)
        input_text.setFont(in_font)

        lv_layout = QVBoxLayout()
        lv_layout.addWidget(input_plot)
        lv_layout.addWidget(input_text)
        lv_layout.addWidget(button_widget)
        lv_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        lv_widget = QWidget()
        lv_widget.setLayout(lv_layout)
        layout_parent.addWidget(lv_widget)

        output_plot = QWidget()

        self.output_figure = plt.figure()
        self.output_canvas = FigureCanvas(self.output_figure)
        output_toolbar = NavigationToolbar(self.output_canvas, output_plot)

        output_plot_layout = QVBoxLayout()
        output_plot_layout.addWidget(output_toolbar)
        output_plot_layout.addWidget(self.output_canvas)
        output_plot.setLayout(output_plot_layout)

        output_text = QLabel("Classifier Output")
        output_text.setAlignment(Qt.AlignHCenter)
        out_font = output_text.font()
        out_font.setPointSize(12)
        output_text.setFont(out_font)

        rv_layout = QVBoxLayout()
        rv_layout.addWidget(output_plot)
        rv_layout.addWidget(output_text)
        rv_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        rv_widget = QWidget()
        rv_widget.setLayout(rv_layout)
        layout_parent.addWidget(rv_widget)

        parent_widget = QWidget()
        parent_widget.setLayout(layout_parent)

        self.setCentralWidget(parent_widget)

        upload.clicked.connect(self.upload)
        download.clicked.connect(self.download)

class Worker(QThread):
    update_status = pyqtSignal(str)

    def __init__(self, address, PP_CHANNELS):
        super().__init__()
        self.address = address
        self.PP_CHANNELS = PP_CHANNELS
        self.continue_run = True

    def run(self):
        # Your long-running task goes here.
        # You can emit signals as needed, for example:
        column_names = ["Time (s)", "Linear Acceleration x (m/s^2)", "Linear Acceleration y (m/s^2)",
                        "Linear Acceleration z (m/s^2)", "Absolute acceleration (m/s^2)"]

        live_df = pd.DataFrame(columns=column_names)

        startTime = time.time()

        print("reached loop")
        while True:
            url = self.address + "/get?" + ("&".join(self.PP_CHANNELS))
            data = requests.get(url=url).json()
            currentTime = time.time()
            accX = data["buffer"][self.PP_CHANNELS[0]]["buffer"][0]
            accY = data["buffer"][self.PP_CHANNELS[1]]["buffer"][0]
            accZ = data["buffer"][self.PP_CHANNELS[2]]["buffer"][0]
            absAcc = np.sqrt(accX ** 2 + accY ** 2 + accZ ** 2)
            live_df.loc[len(live_df.index)] = [currentTime, accX, accY, accZ, absAcc]
            #print([time.time() - startTime, accX, accY, accZ, absAcc])
            if currentTime - startTime >= 5:
                startTime = time.time()
                print(live_df)

                live_sma = live_df.rolling(5).mean().dropna()
                live_sma_arr = [live_sma]
                print(live_sma_arr)
                live_features_df = extract(live_sma_arr)
                print(live_features_df)
                in_norm = trainFit.transform(live_features_df)
                print(in_norm)
                print(l_reg.predict(in_norm)[0])
                status = "Current Status: Walking" if l_reg.predict(in_norm)[0] == 0 else "Current Status: Jumping"
                print(status)
                self.update_status.emit(status)

                live_df = pd.DataFrame(columns=column_names)
class BonusWindow(QWidget):
    def __init__(self):
        super(BonusWindow, self).__init__()
        self.PP_CHANNELS = ["accX","accY","accZ"]
        self.address = None

        label = QLabel("Live Data Evaluation")
        self.con = QLineEdit()
        self.con.setPlaceholderText("Enter address here...")
        self.con.returnPressed.connect(self.return_pressed)
        self.status = QLabel("Current status:")

        parent_layout = QVBoxLayout()
        parent_layout.addWidget(label)
        parent_layout.addWidget(self.status)
        parent_layout.addWidget(self.con)
        self.setLayout(parent_layout)
        self.continue_run = True

    def thread_response(self):
        self.worker = Worker(self.address, self.PP_CHANNELS)
        self.worker.update_status.connect(self.status.setText)
        self.worker.start()

    def return_pressed(self):
        self.address = self.con.text()
        print(self.address)
        self.thread_response()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
