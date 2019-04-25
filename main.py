import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import csv
import os
import glob


def makeDirInDataFolder(dirName):
    '''make a new directory with dirName if it does not exists'''
    if not os.path.exists(os.path.join(os.getcwd(), dirName)):
        os.makedirs(os.path.join(os.getcwd(), dirName))
        print("make ", dirName, " Dir")
    return os.path.join(os.getcwd(), dirName)


def saveToCSV(writeArray, fileName):
    """save a list of row to CSV file"""
    with open(fileName, 'a', newline='') as f:
        csvWriter = csv.writer(f)
        for row in writeArray:
            csvWriter.writerow(row)
    print("save to :" + fileName)


def gauss(x, A, mean, sd):
    return A * np.exp(-1 * ((x - mean) ** 2) / sd ** 2)


def NGauss(numOfGauss, background=True):
    def makeNGauss(xi, *parameters):
        g = 0
        if background:
            gaussParams = parameters[:-1]
            g += parameters[-1]
        else:
            gaussParams = parameters
        for i in range(numOfGauss):
            g += gauss(xi, gaussParams[i * 3], gaussParams[i * 3 + 1], gaussParams[i * 3 + 2])
        return g

    return makeNGauss


def areaUnderGaussian(para):
    A = para[0]
    sd = para[2]
    return np.sqrt(np.pi) * A / (np.sqrt(1 / sd ** 2))


def genIntCond(df, peakLocationArray, intPeakWidth=0.1):
    retrunIntList = []
    for peakLocation in peakLocationArray:
        retrunIntList.append(df.y.values[peakLocation])
        retrunIntList.append(df.x.values[peakLocation])
        retrunIntList.append(intPeakWidth)
    retrunIntList.append(df.y.mean())
    return retrunIntList


def genFittingBound(df, peakLocationArray, boundedWidth=0.1):
    returnUpperBoundList = []
    returnLowerBoundList = []
    for peakLocation in peakLocationArray:
        returnUpperBoundList.append(df.y.values[peakLocation] * 1.001)
        returnUpperBoundList.append(df.x.values[peakLocation] * 1.001)
        returnUpperBoundList.append(boundedWidth * 2.5)

        returnLowerBoundList.append(df.y.values[peakLocation] * 0.7)
        returnLowerBoundList.append(df.x.values[peakLocation] * 0.999)
        returnLowerBoundList.append(boundedWidth * 0.005)

    returnUpperBoundList.append(df.y.mean() * 10)  # df.y.mean()*1.1
    returnLowerBoundList.append(df.y.mean() * 0)

    return [returnLowerBoundList, returnUpperBoundList]


def saveSpectrum(df, popt, peaks, saveFileID):
    style = dict(size=10, color='black')
    peaksFolder = makeDirInDataFolder(saveFileID + "_peaks") + "/"
    plt.figure(figsize=(20, 8))
    plt.plot(df.x, df.y, label="Raw Data")
    plt.scatter(df.x.values[peaks], df.y.values[peaks], marker="x", c="r", label="Identitied Peak")
    for i, val in enumerate(peaks):
        gaussPara = popt[i * 3:i * 3 + 3]
        gaussPara = np.append(gaussPara, popt[-1])
        plt.plot(df.x, NGauss(1)(df.x.values, *gaussPara))
        plt.text(df.x.values[val], df.y.values[val] + df.y.values.max() / 20, '%03d' % i, ha='center', **style)

    addInterestedPeakLine()

    plt.title("Data Set: " + saveFileID)
    plt.legend()
    plt.xlim(df.x.values[1], df.x.values[-1])
    plt.ylim(0, df.y.values.max() * 1.2)
    plt.savefig(peaksFolder + "spectrum.png", dpi=300)
    plt.close()


def savePreFittingSpectrum(df, peaks, saveFileID):
    style = dict(size=10, color='black')
    peaksFolder = makeDirInDataFolder(saveFileID + "_peaks") + "/"
    plt.figure(figsize=(20, 8))
    plt.plot(df.x, df.y, label="Raw Data")
    plt.scatter(df.x.values[peaks], df.y.values[peaks], marker="x", c="r", label="Identitied Peak")

    for i, val in enumerate(peaks):
        plt.text(df.x.values[val], df.y.values[val] + df.y.values.max() / 20, '%03d' % i, ha='center', **style)

    addInterestedPeakLine()

    plt.title("Data Set: " + saveFileID)
    plt.legend()
    plt.xlim(df.x.values[1], df.x.values[-1])
    plt.ylim(0, df.y.values.max() * 1.2)
    plt.savefig(peaksFolder + "preFitting_spectrum.png", dpi=300)
    plt.close()


def saveAllPeaks(df, popt, peaks, saveFileID):
    style = dict(size=15, color='black')
    peaksFolder = makeDirInDataFolder(saveFileID + "_peaks") + "/"
    for idx, val in enumerate(peaks):
        plt.figure(figsize=(8, 6))

        plt.plot(df.x, df.y, label="Raw Data")
        plt.scatter(df.x.values[peaks], df.y.values[peaks], marker="x", c="r", label="Identitied Peak")

        gaussPara = popt[idx * 3:idx * 3 + 3]
        gaussPara = np.append(gaussPara, popt[-1])

        plt.plot(df.x, NGauss(1)(df.x.values, *gaussPara), label="Fitted Peak at x = " + "%.2f" % gaussPara[1])
        plt.text(df.x.values[val], df.y.values[val] * 1.3, '%03d' % idx, ha='center', **style)

        plt.title("Peak " + '%03d' % idx + " at x = " + "%.2f" % gaussPara[1])
        plt.xlim(gaussPara[1] - gaussPara[2] * 50, gaussPara[1] + gaussPara[2] * 50)
        plt.ylim(0, df.y.values[val] * 1.5)

        addInterestedPeakLine()

        plt.legend()
        plt.savefig(peaksFolder + "peak_" + '%03d' % idx + ".png", dpi=200)
        plt.close()

def addInterestedPeakLine():
    for lineXLocation in interestedPeakList:
        plt.axvline(x=lineXLocation)



def readInterestedPeakTxt(filePath):
    with open(filePath) as f:
        data = f.readlines()
    returnList = []
    for item in data:
        print(item.rstrip())
        returnList.append(float(item.rstrip()))
    return returnList


def mainFitting(dataCSVFileName):
    df = pd.read_csv(dataCSVFileName, skiprows=2, usecols=[1, 2], names=["x", "y"])
    saveFileID = os.path.basename(dataCSVFileName)[:-4]
    print(saveFileID)

    print("locating peaks")
    peaks, properties = find_peaks(df.y.values, prominence=[100000, None], height=1000)
    print("peaks found: ", len(peaks))
    print("start fitting")

    popt, pcov = curve_fit(NGauss(len(peaks)), df.x, df.y, p0=genIntCond(df, peaks), bounds=genFittingBound(df, peaks))
    print("finished fitting")

    peaksFolder = makeDirInDataFolder(saveFileID + "_peaks") + "/"
    savePreFittingSpectrum(df, peaks, saveFileID)

    writeCSVArray = []
    for index, val in enumerate(peaks):
        para = popt[index * 3:index * 3 + 3]
        writeCSVArray.append([df.x.values[val], areaUnderGaussian(para)])

    saveToCSV([["X", "Area under peak"]], peaksFolder + saveFileID + "_result.csv")
    saveToCSV(writeCSVArray, peaksFolder + saveFileID + "_result.csv")

    print("Saving Spectrum")
    saveSpectrum(df, popt, peaks, saveFileID)
    print("Saving Peaks")
    saveAllPeaks(df, popt, peaks, saveFileID)


fileList = glob.glob("./*.csv")
fileList = sorted(fileList)
interestedPeakList = readInterestedPeakTxt("./interestedPeakList.txt")

for index, fileName in enumerate(fileList):

    print("-----------", "Fitting:", fileName, ",", index + 1, "/", len(fileList), "-----------")
    try:
        mainFitting(fileName)
    except ValueError:
        print("ValueError: " + fileName)
        valueErrorFolder = makeDirInDataFolder("ValueError")
        newPath = os.path.join(valueErrorFolder, fileName)
        os.rename(fileName, newPath)
        print("relocate file to: " + newPath)

print("Finished")
