from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import csv
import os
import math
import time

#################### global variables ####################
fuzzyPath = "./fuzzydatasets/fuzzydata.csv"
csvFilePath = "./datasets/1/name.csv"
originalDatasetPath = "./fuzzydatasets/originaldata.csv" 

########### csv functions ####################

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    your_list = [item for sublist in your_list for item in sublist]
    your_list = [x.lower().replace('"',  '') for x in your_list]
    #replace all quotes with nothing
    your_list = [x.replace("'",  '') for x in your_list]
    #replace all double quotes with nothing
    your_list = [x.replace('"',  '') for x in your_list]
    #replace all . with nothing
    your_list = [x.replace('.',  '') for x in your_list]
    #remove all , from the list
    your_list = [x.replace(',',  '') for x in your_list]
    
    return your_list

#function that takes in a csv file and a list of strings
# the function appends the list of strings to the csv file
# the function returns nothing
def append_to_csv(list, path = fuzzyPath):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #write each string in the list to a new line in the csv file
        for item in list:
            writer.writerow([item])
    
    return None

#function that takes in n number of lines from a csv file
# and returns a list of strings from the csv file
def get_csv_lines_randomly(n):
    csvLines = load_csv(csvFilePath)
    #choose "n" random lines from the csv file
    csvLines = random.sample(csvLines, n)
    
    return csvLines

#a fucntion to go through "./fuzzydatasets/" and load each csv file into a list
def collect_datasets(path = "./fuzzydatasets/"):
    #create a list of all the csv files in the fuzzydatasets directory
    csv_files = [pos_csv for pos_csv in os.listdir(path) if pos_csv.endswith('.csv')]
    #create a dictionary of lists, where the key is the name of the csv file and the value is the list of strings
    csv_dict = {}
    for csv_file in csv_files:
        #create a list of strings from the csv file
        csv_list = load_csv(path + csv_file)
        #add the list to the dictionary with the key being the name of the csv file
        csv_dict[csv_file] = csv_list
    return csv_dict

#Function that takes in a file name and a list of floats and a list of ints
#The function will create a csv file with the name of the file name
#The function will write each float in the list to a new line in the csv file
#The function will write each int in the list next to the corresponding float
#The function will return the saved csv file path
def save_csv_file(file_name, nStrings, timeTaken):
    #create a path to save the csv file
    path = "./results/" + file_name + ".csv"
    # if the file does not exist, create it
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            #write the headers to the csv file
            writer.writerow(["nStrings", "Training Time"])
    
    #open the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #write each float and int to a new line in the csv file
        writer.writerow([nStrings, timeTaken])
    return path

########## end csv functions #################


############# Main #####################
if __name__ == '__main__':
    
    for csv_file in os.listdir("./datasets/"):
        #if file is not csv skip
        if not csv_file.endswith('.csv'):
            continue
        
        original_list = load_csv(f"./datasets/{csv_file}")

        model = PolyFuzz("TF-IDF")
        
        #start the timer
        start = time.time()
        
        model.fit(original_list)
        model.save(f"./models/{len(original_list)}_model_train")
        
        end = time.time()
        
        #calculate the time taken to train the model
        timeTaken = end - start
        #save the csv file
        save_csv_file("Model_Training", len(original_list), timeTaken)
 
    

from_list = load_csv("/media/max/2AB8BBD1B8BB99B1/MntStn/Apps/Collection/src/resources/2020/1/lookup/name.csv")
#print the first 5 elements of the list
print(from_list[:5])
to_list = ["LAUDER LEONARD", "LANDEC CORP"]

model = PolyFuzz("TF-IDF")
model.fit(from_list)
# Save the model
model.save("./models/my_model")

#result = model.transform(to_list)

#model = PolyFuzz("TF-IDF").match(from_list, to_list)
#print(model.get_matches())
#print(result)