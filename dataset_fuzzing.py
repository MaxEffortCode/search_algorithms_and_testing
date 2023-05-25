from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import sys
import os
import csv
import random
from string import ascii_lowercase
import copy
import numpy as np

#Global variables
fuzzyPath = "./fuzzydatasets/fuzzydata.csv"
csvFilePath = "./datasets/1/name.csv"

###################  edit distance functions ############################

def char_edit_distance(c1, c2):

    # Define the keyboard layout
    keyboard = [
        ['`','1','2','3','4','5','6','7','8','9','0','-','='],
        ['\t','q','w','e','r','t','y','u','i','o','p','[',']','\\'],
        ['a','s','d','f','g','h','j','k','l',';','\''],
        ['z','x','c','v','b','n','m',',','.','/']
    ]
    
    # Get the positions of the two characters on the keyboard
    pos1, pos2 = None, None
    for row_idx, row in enumerate(keyboard):
        if c1 in row:
            pos1 = (row_idx, row.index(c1))
        if c2 in row:
            pos2 = (row_idx, row.index(c2))
        if pos1 is not None and pos2 is not None:
            break
    
    # If either character is not found on the keyboard, return a very large distance
    if pos1 is None or pos2 is None:
        return 999
    
    # Calculate the distance between the characters based on their keyboard positions
    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    return dist

def random_edit(string):
    # Choose a random character in the string
    try:
        index = random.randint(0, len(string) - 1)
        char = string[index]

        # Generate a list of possible replacement characters
        alphabet = ascii_lowercase
        candidates = [c for c in alphabet if char_edit_distance(char, c) <= 1 and c != char]
        #print(f"candidates: {candidates}")

        # Choose a replacement character at random
        if candidates:
            replacement = random.choice(candidates)
            edited_string = string[:index] + replacement + string[index+1:]
            return edited_string
        else:
            return string
    except:
        return string

#the function randomly swaps a vowel in the string with another vowel
def random_vowel_swap(string):
    #create a list of vowels
    vowels = ['a', 'e', 'i', 'o', 'u']
    #create a list of the indices of the vowels in the string
    vowel_indices = [i for i, letter in enumerate(string) if letter in vowels]
    #choose a random index from the list of vowel indices
    try:
        index = random.choice(vowel_indices)
    except:
        return string
    #choose a random vowel from the list of vowels
    vowel = random.choice(vowels)
    #replace the vowel at the random index with the random vowel
    string = string[:index] + vowel + string[index+1:]
    
    return string


######## end edit distance functions #########

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
    #remove the first element of the list, which is the column header
    your_list.pop(0)
    
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
def get_csv_lines_randomly(n, path = csvFilePath):
    csvLines = load_csv(path)
    #choose "n" random lines from the csv file
    csvLines = random.sample(csvLines, n)
    
    return csvLines


########## end csv functions #################


######### fuzzing functions ##################

def fuzz_list_edit_distance_substitution(list):
    fuzzed_list = []
    for item in list:
        item = random_edit(item)
        fuzzed_list.append(item)
        
    return fuzzed_list

def fuzz_list_vowel_swap_substitution(list):
    fuzzed_list = []
    for item in list:
        item = random_vowel_swap(item)
        fuzzed_list.append(item)
        
    return fuzzed_list

def fuzz_list_ommission(list):
    fuzzed_list = []
    for item in list:
        #choose a random index in the string
        index = random.randint(0, len(item) - 1)
        #remove the character at the random index
        item = item[:index] + item[index+1:]
        fuzzed_list.append(item)
        
    return fuzzed_list

def fuzz_list_transpose(list):
    fuzzed_list = []
    for item in list:
        #choose a random index in the string
        index = random.randint(0, len(item) - 2)
        #swap the characters at the random index and the index after it
        item = item[:index] + item[index+1] + item[index] + item[index+2:]
        fuzzed_list.append(item)
        
    return fuzzed_list

#function to add an insertion to a string
# the insertion is a a charcter with a random edit distance of 1
def fuzz_list_insertion(list):
    fuzzed_list = []
    for item in list:
        #choose a random index in the string
        index = random.randint(0, len(item) - 1)
        #choose a random character from the alphabet
        alphabet = ascii_lowercase
        char = random.choice(alphabet)
        #insert the random character at the random index
        item = item[:index] + char + item[index:]
        fuzzed_list.append(item)
        
    return fuzzed_list


if __name__ == '__main__':
    #create a list of strings from the csv file
    # the list is the first 100 lines of the csv file
    # then the list is fuzzed
    # then the fuzzed list is appended to the csv file
    
    nLines = 5000
    for year in range(1994,2023):
        for qtr in range(1,5):
            
            print(f"nLines: {nLines}")
            print(f" getting lines from ./resources/{year}/{qtr}/lookup/name.csv")
            lines = get_csv_lines_randomly(nLines, path=f"./resources/{year}/{qtr}/lookup/name.csv")
        
            append_to_csv(lines, path=f"./resources/{year}/{qtr}/lookup/test_data_original.csv")
            print(f"original lines: \n {lines}\n")
            
            fuzzed_lines_edit_distance = fuzz_list_edit_distance_substitution(lines)
            append_to_csv(fuzzed_lines_edit_distance, path=f"./resources/{year}/{qtr}/lookup/substitution_test.csv")
            print(f"fuzzed_lines_edit_distance: \n {fuzzed_lines_edit_distance}\n")
            
            fuzzed_lines_vowel_swap = fuzz_list_vowel_swap_substitution(lines)
            append_to_csv(fuzzed_lines_vowel_swap, path=f"./resources/{year}/{qtr}/lookup/swap_test.csv")
            print(f"fuzzed_lines_vowel_swap: \n {fuzzed_lines_vowel_swap}\n")
            
            fuzzed_lines_ommission = fuzz_list_ommission(lines)
            append_to_csv(fuzzed_lines_ommission, path=f"./resources/{year}/{qtr}/lookup/ommision_test.csv")
            print(f"fuzzed_lines_ommission: \n {fuzzed_lines_ommission}\n")
            
            fuzzed_lines_transpose = fuzz_list_transpose(lines)
            append_to_csv(fuzzed_lines_transpose, path=f"./resources/{year}/{qtr}/lookup/transposition_test.csv")
            print(f"fuzzed_lines_transpose: \n {fuzzed_lines_transpose}\n")
            
            fuzzed_lines_insertion = fuzz_list_insertion(lines)
            append_to_csv(fuzzed_lines_insertion, path=f"./resources/{year}/{qtr}/lookup/inserstion_test.csv")
            print(f"fuzzed_lines_insertion: \n {fuzzed_lines_insertion}\n")