from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import random
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
def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken, csv_path = "./results/"):
    #create a path to save the csv file
    path = "./results/" + file_name + ".csv"
    # if the file does not exist, create it
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            #write the headers to the csv file
            writer.writerow(["num_correct", "num_incorrect", "num_unmatched", "num_total", "time_taken", "type of test"])
    
    #open the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #write each float and int to a new line in the csv file
        writer.writerow([num_correct, num_incorrect, num_unmatched, num_total, time_taken, csv_path])
    return path


########## end csv functions #################


############# Main #####################
if __name__ == '__main__':
    for year in range(1993, 2023):
        for qrt in range(1,5):
            path=f"./resources/{year}/{qrt}/lookup/"
            for csv_file in os.listdir(path):
                if csv_file.endswith('.csv') and not csv_file.__contains__("name"):
                    fuzzed_list = load_csv(path + csv_file)
                    original_list = load_csv(f"./resources/{year}/{qrt}/lookup/test_data_original.csv")
                    model = PolyFuzz.load(f"./resources/{year}/{qrt}/models/c_name_model")
                    
                    start = time.time()
                    correct, incorrect, itemNum = 0, 0, 0
                    for item in fuzzed_list:
                        to_list = [item]
                        result = model.transform(to_list)
                        
                        if result['TF-IDF'].values[0][1] == original_list[itemNum]:
                            correct += 1
                        else:
                            incorrect += 1
                        
                        itemNum += 1
                        if time.time() - start > 100:
                            print("Time limit reached")
                            break
                        
                    end = time.time()

                    print(f"trained model Time to run {csv_file}: {end - start}")
                    print(f"Correct: {correct}\nIncorrect: {incorrect}")
                    #print the average time of each search
                    print(f"Average time of each search: {(end - start) / len(fuzzed_list)}")
                    #print the percentage of correct matches
                    print(f"Percentage of correct matches: {correct / (correct + incorrect)}\n")
                    #print the number found vs the number not searched for because of time limit
                    print(f"Number found: {correct}\nNumber not searched for: {len(fuzzed_list) - itemNum}\n")
                    #use the save_csv_file function to save the results to a csv file
                    #def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken):
                    save_csv_file("trained_model_results", correct, incorrect, len(fuzzed_list) - itemNum, len(fuzzed_list), end - start, csv_file.split(".")[:-1])

