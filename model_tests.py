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
def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken):
    #create a path to save the csv file
    path = "./results/" + file_name + ".csv"
    # if the file does not exist, create it
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            #write the headers to the csv file
            writer.writerow(["num_correct", "num_incorrect", "num_unmatched", "num_total", "time_taken"])
    
    #open the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        #write each float and int to a new line in the csv file
        writer.writerow([num_correct, num_incorrect, num_unmatched, num_total, time_taken])
    return path


########## end csv functions #################


############# Main #####################
if __name__ == '__main__':
    
    original_lists_paths = []
    models_paths = []
    n_search_terms = []
    time_to_search_per_n = []

    for csv_file in os.listdir("./models/"):
        #get all models
        models_paths.append("./models/" + csv_file)
    
    for csv_file in os.listdir("./datasets/"):
        #if not a csv file, skip it
        if not csv_file.endswith('.csv'):
            continue
        #get all models
        original_lists_paths.append("./datasets/" + csv_file)

    #sort the list of paths
    original_lists_paths.sort()
    
    print("Original Lists Paths: ", original_lists_paths)
    print("Models Paths: ", models_paths)
    
    i = 0
    for csv_file in os.listdir("./fuzzydatasets/"):
        #if the csv file conatins the number "2" in it then skip it

        #print what i is
        print("i: ", i)
        #print math.floor(i/5)
        print("math.floor(i/5): ", math.floor(i/5))
        original_list = load_csv(original_lists_paths[math.floor(i/5)])
        models_path = models_paths[math.floor(i/5)]
        #create a list of strings from the csv file
        
        loaded_model = PolyFuzz.load(models_path)
        
        fuzzed_list = load_csv("./fuzzydatasets/" + csv_file)
        
        #start the timer
        start = time.time()
        
        #create a for loop that goes through each item in the csv list
        itemNum = 0
        correct, incorrect = 0, 0
        for item in fuzzed_list:
            to_list = [item]
            result = loaded_model.transform(to_list)
            #print(f"\n {result['TF-IDF'].values[0][1]}\n")
            
            #for key in result.keys():
                #print(f"key: {key}")
            
            #print(result)
            #update the search term
            #check if the search term was found
            #print(f"Searching for {original_list[itemNum]} in {csv_file}")
            #time.sleep(1)
            
            #check if the result is equal to the original list item
            #check if the "From" value in result is equal to the original list item
            if result['TF-IDF'].values[0][1] == original_list[itemNum]:
                #print(f"Found {result} in {csv_file}")
                correct += 1

            else:
                incorrect += 1
            #update the item number
            itemNum += 1
            
            #if start time minus current time is greater than 100 seconds, break
            if time.time() - start > 100:
                print("Time limit reached")
                break
        #end the timer
        end = time.time()
        #print the time it took to run the search
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
        save_csv_file("trained_model_results", correct, incorrect, len(fuzzed_list) - itemNum, len(fuzzed_list), end - start)
        i += 1
