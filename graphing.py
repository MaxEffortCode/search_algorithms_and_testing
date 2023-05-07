#import all the necessary libraries for graphing csv files
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import math
import time
from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


#function that takes in a csv and a int that is a column number
# the function returns a list of strings from the csv file based on the column number
def load_csv(path, column):
    your_list = []
    #check if the path is a csv file
    if not path.endswith('.csv'):
        print("error: path is not a csv file")
        return None
    
    #set your_list to contain all the selected column values
    with open(path, 'r') as f:
        reader = csv.reader(f)
        #skip the first row
        next(reader)
        for row in reader:
            your_list.append((float(row[column])))
    
    your_list.pop(0)
    print(f"your_list: {your_list}")
    
    #go through the list in sets of 5 and average the values
    # Then append the averaged values to a new list
    newList = []
    for i in range(0, len(your_list), 5):
        print(i)
        print(your_list[i:i+5])
        print(sum(your_list[i:i+5]))
        print(sum(your_list[i:i+5])/5)
        newList.append(sum(your_list[i:i+5])/5)

    
    
    return newList


#function that creates a graph from two lists
# the function also takes in a save path for the graph
# the function returns the save path
def create_graph(title, x, y, xLabel, yLabel, savePath):
    #create a new figure
    plt.figure()
    #give the figure a title
    plt.title(title)
    #set the x and y labels
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    #create a second x axis
    #plot the x and y values
    plt.plot(x, y)
    #match an equation to the graph
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    #create a key for the graph
    plt.legend(['data', 'linear fit'], loc='best')
        
    #save the figure to the save path
    plt.savefig(savePath)
    
    return savePath


if __name__ == '__main__':
    searchAlgo = "TF-IDF Model"
    algoPath = "results/trained_model_results.csv"
    #num_correct,num_incorrect,num_unmatched,num_total,time_taken
    numCorect = load_csv(algoPath, 0)
    numIncorrect = load_csv(algoPath, 1)
    numTotal = load_csv(algoPath, 3)
    timeTaken = load_csv(algoPath, 4)
    
    #create a list that is the ratio of correct to incorrect
    percentCorrect = []
    for i in range(len(numCorect)):
        percentCorrect.append(numCorect[i] / (numCorect[i]+numIncorrect[i]))
        
    
    
    #create a list of average time taken per search term
    timeTakenPerSearchTerm = []
    for i in range(len(timeTaken)):
        timeTakenPerSearchTerm.append(timeTaken[i] / (numCorect[i]+numIncorrect[i]))
    
    
    
    print(f"numTotal: {numTotal}")
    create_graph( f"{searchAlgo} Accuracy", numTotal, percentCorrect,  "DATA SET SIZE", "PERCENT CORRECT", f"./graphs/{searchAlgo}_accuracy.png")
    create_graph( f"{searchAlgo} Time" , numTotal, timeTakenPerSearchTerm,  "DATA SET SIZE", "RUN-TIME PER SEARCH TERM", f"./graphs/{searchAlgo}_time.png")