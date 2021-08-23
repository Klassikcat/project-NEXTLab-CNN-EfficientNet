import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Visualization:
    def __init__(self, dataframe):
        """
        dataframe = type:pd.DataFrame, full dataframe that contains information.
        """
        self.dataframe = dataframe

    def counts(self, name=str):
        return self.dataframe[name].value_counts()

    def cutName(self, data, threshold=int):
        self.data = data
        dataframe = pd.DataFrame(data).transpose()
        others = 0
        for i in dataframe.columns.tolist():
            if int(dataframe[i]) < threshold:
                others += int(dataframe[i])
                dataframe = dataframe.drop(i, axis=1)
        dataframe['기타'] = others
        return dataframe.transpose()

    def pieplot(self, dataframe):
        self.dataframe = dataframe;
        explode = [0.10] * len(dataframe.index.tolist());

        plt.figure(figsize=(15, 15));
        plt.pie(dataframe, labels=dataframe.index.tolist(), autopct='%.1f%%', startangle=260, counterclock=False,
                explode=explode);
        plt.show();

    def plot(self, name=str, threshold=int):
        self.name = name
        self.threshold = threshold

        colDataFrame = self.counts(name)
        cuttedDataFrame = self.cutName(colDataFrame, threshold)
        self.pieplot(cuttedDataFrame)