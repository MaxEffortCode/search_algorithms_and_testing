import os
from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import sys
import random
import csv
import math
import time
import numpy as np

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
    path = "./results/" + file_name
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


############ Levenshtein Clases ####################
class LevenshteinSearcher:
    def __init__(self, array, search_term):
        self.array = array
        self.search_term = search_term
        
    def update_array(self, array):
        self.array = array
        
    def update_search_term(self, search_term):
        self.search_term = search_term

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def search(self):
        #if the search term is a CIK
        if self.search_term.isdigit():
            #if the search term is in the array
            if self.search_term in self.array:
                #return the search term
                return self.search_term
            #if the search term is not in the array
            else:
                #return none
                return None
        #if the search term is not a CIK
        else:
            #return the closest match
            return self.closest_match()

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def closest_match(self):
        #time the different methods
        #time the levenshtein distance method
        closest_match = ""
        start = time.time()
        closest_match = self.closest_match_levenshtein()
        end = time.time()
        
        
        return closest_match

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def closest_match_levenshtein(self):
        #set the closest match to none
        closest_match = None
        #set the closest match distance to the length of the search term
        closest_match_distance = len(self.search_term)
        #for each item in the array
        for item in self.array:
            #set the distance to the levenshtein distance between the search term and the item
            distance = self.levenshtein_distance(self.search_term, item)
            #if the distance is less than the closest match distance
            if distance < closest_match_distance:
                #set the closest match to the item
                closest_match = item
                #set the closest match distance to the distance
                closest_match_distance = distance
        #return the closest match
        return closest_match

    #function that takes in self, a, and b
    #it will return the levenshtein distance between a and b
    def levenshtein_distance(self, a, b):
        #if a is empty
        if len(a) == 0:
            #return the length of b
            return len(b)
        #if b is empty
        if len(b) == 0:
            #return the length of a
            return len(a)
        #set the first row to the length of a
        first_row = range(len(a) + 1)
        #for each item in b
        for i, c1 in enumerate(b):
            #set the second row to the length of a
            second_row = [i + 1]
            #for each item in a
            for j, c2 in enumerate(a):
                #if the item in a is equal to the item in b
                if c1 == c2:
                    #set the second row to the first row
                    second_row.append(first_row[j])
                #if the item in a is not equal to the item in b
                else:
                    #set the second row to the minimum of the first row, the second row, and the first row
                    second_row.append(1 + min((first_row[j], first_row[j + 1], second_row[-1])))
                
            #set the first row to the second row
            first_row = second_row
        #return the last item in the first row
        return first_row[-1]
        
############ end Levenshtein Class ####################

############ Levenshtein_Non_naive Clases ####################
class LevenshteinSearcherNotNaive:
    def __init__(self, array, search_term):
        self.array = array
        self.search_term = search_term
        
    def update_array(self, array):
        self.array = array
        
    def update_search_term(self, search_term):
        self.search_term = search_term

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def search(self):
        #if the search term is a CIK
        if self.search_term.isdigit():
            #if the search term is in the array
            if self.search_term in self.array:
                #return the search term
                return self.search_term
            #if the search term is not in the array
            else:
                #return none
                return None
        #if the search term is not a CIK
        else:
            #return the closest match
            return self.closest_match()

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def closest_match(self):
        #time the different methods
        #time the levenshtein distance method
        closest_match = ""
        start = time.time()
        closest_match = self.closest_match_levenshtein()
        end = time.time()
        
        
        return closest_match

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def closest_match_levenshtein(self):
        #set the closest match to none
        closest_match = None
        #set the closest match distance to the length of the search term
        closest_match_distance = len(self.search_term)
        #for each item in the array
        for item in self.array:
            #set the distance to the levenshtein distance between the search term and the item
            distance = self.levenshtein_distance(self.search_term, item)
            #if the distance is less than the closest match distance
            if distance < closest_match_distance:
                #set the closest match to the item
                closest_match = item
                #set the closest match distance to the distance
                closest_match_distance = distance
        #return the closest match
        return closest_match

    #function that takes in self, a, and b
    #it will return the levenshtein distance between a and b
    def levenshtein_distance(self, a, b, threshold=float(30)):
        # if a is empty
        if len(a) == 0:
            # return the length of b
            return len(b)
        # if b is empty
        if len(b) == 0:
            # return the length of a
            return len(a)
        # if the difference in length of a and b is greater than the threshold
        if abs(len(a) - len(b)) > threshold:
            # return a large value to indicate very dissimilar strings
            return float('inf')
        # set the first row to the length of a
        first_row = range(len(a) + 1)
        # for each item in b
        for i, c1 in enumerate(b):
            # set the second row to the length of a
            second_row = [i + 1]
            # for each item in a
            for j, c2 in enumerate(a):
                # if the item in a is equal to the item in b
                if c1 == c2:
                    # set the second row to the first row
                    second_row.append(first_row[j])
                # if the item in a is not equal to the item in b
                else:
                    # set the second row to the minimum of the first row, the second row, and the first row
                    second_row.append(1 + min((first_row[j], first_row[j + 1], second_row[-1])))
            
            # set the first row to the second row
            first_row = second_row
            # if the last item in the first row exceeds the threshold
            if first_row[-1] > threshold:
                # return a large value to indicate very dissimilar strings
                return float('inf')
        # return the last item in the first row
        return first_row[-1]
        
############ end Levenshtein Class ####################

############# begin Smith-Waterman algorithm ##############
class SmithWatermanSearcher:
    def __init__(self, array, search_term):
        self.array = array
        self.search_term = search_term
        
    def update_array(self, array):
        self.array = array
        
    def update_search_term(self, search_term):
        self.search_term = search_term

    #function that takes in self
    #it will return the closest match to the search term
    #it will have a variety of methods to find the closest match
    def search(self):
        scores = []
        for seq in self.array:
            # initialize the score matrix with zeros
            score_matrix = np.zeros((len(self.search_term) + 1, len(seq) + 1))

            # compute the score matrix using dynamic programming
            for i in range(1, len(self.search_term) + 1):
                for j in range(1, len(seq) + 1):
                    match_score = 2 if self.search_term[i-1] == seq[j-1] else -1
                    score_matrix[i][j] = max(0, 
                        score_matrix[i-1][j-1] + match_score, 
                        score_matrix[i-1][j] - 1, 
                        score_matrix[i][j-1] - 1)
            
            # find the highest score and its position in the matrix
            max_score = np.max(score_matrix)
            max_pos = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

            # backtrack from the highest score to get the aligned sequences
            aligned_query = ''
            aligned_seq = ''
            i, j = max_pos
            while score_matrix[i][j] > 0:
                if score_matrix[i-1][j-1] + (2 if self.search_term[i-1] == seq[j-1] else -1) == score_matrix[i][j]:
                    aligned_query = self.search_term[i-1] + aligned_query
                    aligned_seq = seq[j-1] + aligned_seq
                    i -= 1
                    j -= 1
                elif score_matrix[i-1][j] - 1 == score_matrix[i][j]:
                    aligned_query = self.search_term[i-1] + aligned_query
                    aligned_seq = '-' + aligned_seq
                    i -= 1
                elif score_matrix[i][j-1] - 1 == score_matrix[i][j]:
                    aligned_query = '-' + aligned_query
                    aligned_seq = seq[j-1] + aligned_seq
                    j -= 1
            
            # store the score and aligned sequences for this sequence
            scores.append((max_score, aligned_query, aligned_seq))

        #get the closest match
        closest_match = max(scores, key=lambda x: x[0])
        closest_match = closest_match[2]
        # return the sequence with the highest score
        return closest_match

############ end Smith-Waterman algorithm ####################





############# Main #####################
if __name__ == '__main__':
    

    original_lists_paths = []
    n_search_terms = []
    time_to_search_per_n = []

    for csv_file in os.listdir("./datasets/"):
        #skip any fodlers
        if csv_file.endswith(".csv"):
            #add the path to the csv file to the list of paths
            original_lists_paths.append("./datasets/" + csv_file)
    
    #sort the list of paths
    original_lists_paths.sort()
    
    print("Original Lists Paths: ", original_lists_paths)
    
    i = 0
    for csv_file in os.listdir("./fuzzydatasets/"):
        #if the csv file conatins the number "2" in it then skip it

        #print what i is
        print("i: ", i)
        #print math.floor(i/5)
        print("math.floor(i/5): ", math.floor(i/5))
        original_list = load_csv(original_lists_paths[math.floor(i/5)])
        #create a list of strings from the csv file
        fuzzed_list = load_csv("./fuzzydatasets/" + csv_file)
        #start the timer
        start = time.time()
        #create a LevenshteinSearcher object
        LevenshteinSearch = LevenshteinSearcherNotNaive(original_list, "None")
        #create a for loop that goes through each item in the csv list
        itemNum = 0
        correct, incorrect = 0, 0
        for item in fuzzed_list:
            #update the search term
            LevenshteinSearch.update_search_term(item)
            #run the search method
            found_term = LevenshteinSearch.search()
            #check if the search term was found
            if found_term == original_list[itemNum]:
                print(f"Found {found_term} in {csv_file}")
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
        print(f"LevenshteinSearcher_non_naive Time to run {csv_file}: {end - start}")
        print(f"Correct: {correct}\nIncorrect: {incorrect}")
        #print the average time of each search
        print(f"Average time of each search: {(end - start) / len(fuzzed_list)}")
        #print the percentage of correct matches
        print(f"Percentage of correct matches: {correct / (correct + incorrect)}\n")
        #print the number found vs the number not searched for because of time limit
        print(f"Number found: {correct}\nNumber not searched for: {len(fuzzed_list) - itemNum}\n")
        #use the save_csv_file function to save the results to a csv file
        #def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken):
        save_csv_file(f"Levenshtein_results_non_naive", correct, incorrect, len(fuzzed_list) - itemNum, len(fuzzed_list), end - start)
        i += 1
        
    
    i = 0
    for csv_file in os.listdir("./fuzzydatasets/"):
        original_list = load_csv(original_lists_paths[math.floor(i/5)])
        #create a list of strings from the csv file
        fuzzed_list = load_csv("./fuzzydatasets/" + csv_file)
        #start the timer
        start = time.time()
        #create a LevenshteinSearcher object
        LevenshteinSearch = LevenshteinSearcher(original_list, "None")
        #create a for loop that goes through each item in the csv list
        itemNum = 0
        correct, incorrect = 0, 0
        for item in fuzzed_list:
            #update the search term
            LevenshteinSearch.update_search_term(item)
            #run the search method
            found_term = LevenshteinSearch.search()
            #check if the search term was found
            if found_term == original_list[itemNum]:
                print(f"Found {found_term} in {csv_file}")
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
        print(f"LevenshteinSearcher Time to run {csv_file}: {end - start}")
        print(f"Correct: {correct}\nIncorrect: {incorrect}")
        #print the average time of each search
        print(f"Average time of each search: {(end - start) / len(fuzzed_list)}")
        #print the percentage of correct matches
        print(f"Percentage of correct matches: {correct / (correct + incorrect)}\n")
        #print the number found vs the number not searched for because of time limit
        print(f"Number found: {correct}\nNumber not searched for: {len(fuzzed_list) - itemNum}")
        #use the save_csv_file function to save the results to a csv file
        #def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken):
        save_csv_file(f"Levenshtein_results", correct, incorrect, len(fuzzed_list) - itemNum, len(fuzzed_list), end - start)
        i += 1
    
    
    i = 0
    for csv_file in os.listdir("./fuzzydatasets/"):
        original_list = load_csv(original_lists_paths[math.floor(i/5)])
        #create a list of strings from the csv file
        fuzzed_list = load_csv("./fuzzydatasets/" + csv_file)
        #start the timer
        start = time.time()
        #create a LevenshteinSearcher object
        SmithWatermanSearch = SmithWatermanSearcher(original_list, "None")
        #create a for loop that goes through each item in the csv list
        itemNum = 0
        correct, incorrect = 0, 0
        for item in fuzzed_list:
            #update the search term
            SmithWatermanSearch.update_search_term(item)
            #run the search method
            found_term = SmithWatermanSearch.search()
            #check if the search term was found
            if found_term == original_list[itemNum]:
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
        print(f"SmithWatermanSearcher Time to run {csv_file}: {end - start}")
        print(f"Correct: {correct}\nIncorrect: {incorrect}")
        #print the average time of each search
        print(f"Average time of each search: {(end - start) / len(fuzzed_list)}")
        #print the percentage of correct matches
        print(f"Percentage of correct matches: {correct / (correct + incorrect)}\n")
        #use the save_csv_file function to save the results to a csv file
        #def save_csv_file(file_name, num_correct, num_incorrect, num_unmatched, num_total, time_taken):
        save_csv_file(f"Smith_Waterman_results", correct, incorrect, len(fuzzed_list) - itemNum, len(fuzzed_list), end - start)
        i += 1


    