from polyfuzz import PolyFuzz
from polyfuzz.models import EditDistance, TFIDF, Embeddings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import csv
import os
import random
from flair.embeddings import TransformerWordEmbeddings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz.models import BaseMatcher

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

class CustomTFIDF(TFIDF):
    def __init__(self, n_gram_range=(1, 1), clean_string=True, min_similarity=0.8, top_n=1, cosine_method="knn", model_id="Custom-TF-IDF"):
        super().__init__(n_gram_range, clean_string, min_similarity, top_n, cosine_method, model_id)
    
    def cosine(self, matrix1, matrix2):
        if self.cosine_method == "knn":
            topn = matrix2.shape[0] if matrix2.shape[0] < self.top_n else self.top_n
            similarity_matrix = sparse_cosine_similarity(matrix1, matrix2, topn)
        else:
            similarity_matrix = super().cosine(matrix1, matrix2)

        return similarity_matrix


def create_model_TF_IDF(year, qrt):
    print(f"Creating TF-IDF model for year: {year} Quarter: {qrt}")
    csv_file = f"./resources/{year}/{qrt}/lookup/name.csv"
    company_list = load_csv(csv_file)
    model = PolyFuzz("TF-IDF")
    model.fit(company_list)
    # Make a directory to save the model
    os.makedirs(f"./resources/{year}/{qrt}/models", exist_ok=True)
    model.save(f"./resources/{year}/{qrt}/models/c_name_model")
    # Delete the model to free up memory
    del model
    # Delete the old model file if it exists
    if os.path.exists(f"./resources/{year}/{qrt}/comp_name_model"):
        os.remove(f"./resources/{year}/{qrt}/comp_name_model")
    print(f"Model created for year: {year} Quarter: {qrt}")

def create_model_TF_IDF_custom(year, qrt):
    print(f"Creating TF-IDF model for year: {year} Quarter: {qrt}")
    csv_file = f"./resources/{year}/{qrt}/lookup/name.csv"
    company_list = load_csv(csv_file)

    #changed ngram from 1 to 2
    #tfidf = TFIDF(n_gram_range=(2, 2), clean_string=False, min_similarity=0.99, top_n=1, cosine_method="sparse", model_id="Custom-TF-IDF")
    tfidf = CustomTFIDF(n_gram_range=(2, 2), clean_string=False, min_similarity=0.99, top_n=1, cosine_method="knn", model_id="Custom-TF-IDF")
    model = PolyFuzz(tfidf)
    print(f"Model {model}")
    #model.match(company_list, to_list=None)
    model.fit(company_list)
    # Make a directory to save the model
    os.makedirs(f"./resources/{year}/{qrt}/models", exist_ok=True)
    model.save(f"./resources/{year}/{qrt}/models/c_name_model_custom")
    # Delete the model to free up memory
    del model
    # Delete the old model file if it exists
    if os.path.exists(f"./resources/{year}/{qrt}/comp_name_model"):
        os.remove(f"./resources/{year}/{qrt}/comp_name_model")
    print(f"Model created for year: {year} Quarter: {qrt}")


#slow af 
def create_model_embeddings(year, qrt):
    print(f"Creating model for year: {year} Quarter: {qrt}")
    csv_file = f"./resources/{year}/{qrt}/lookup/name.csv"
    company_list = load_csv(csv_file)
    
    embeddings = TransformerWordEmbeddings('bert-base-multilingual-cased')
    print(f"Embeddings loaded: {embeddings}")
    bert = Embeddings(embeddings, min_similarity=0.5, model_id="bert")
    print("BERT loaded")
    tfidf = TFIDF(min_similarity=0)
    print("TF-IDF loaded")
    edit = EditDistance(n_jobs=-1, scorer="fuzz_qratio")
    print("Edit Distance loaded")
    string_models = [bert]
    model = PolyFuzz(bert)
    print("Model created")
    model.fit(company_list)
    print("Model fitted")

    # Make a directory to save the model
    os.makedirs(f"./resources/{year}/{qrt}/models", exist_ok=True)
    model.save(f"./resources/{year}/{qrt}/models/c_name_model_flair")
    print("Model saved")

    del model
    
    print(f"Model created for year: {year} Quarter: {qrt}")





############# Main #####################
if __name__ == '__main__':
    threads = []
    #create_model_embeddings(1996, 1)
    #create_model_TF_IDF(1996, 1)
    create_model_TF_IDF_custom(1996, 1)


    """ with ThreadPoolExecutor(max_workers=4) as executor:
        for year in range(1993, 2023):
            for qrt in range(1,5):
                executor.submit(create_model_flair, year, qrt) """
