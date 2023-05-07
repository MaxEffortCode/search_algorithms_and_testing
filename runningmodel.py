from polyfuzz import PolyFuzz
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import time

# Load the model
loaded_model = PolyFuzz.load("./models/my_model")


while True:    
    userinput = input("enter company name: ")
    #user input to lowercase because the model was trained on lowercase
    userinput = userinput.lower()
    start = time.time()
    to_list = [userinput]
    result = loaded_model.transform(to_list)
    print(result)
    end = time.time()
    print(f"search time elapsed: {end - start}")


'''
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from polyfuzz.models import BaseMatcher


class MyModel(BaseMatcher):
    def match(self, from_list, to_list, **kwargs):
        # Calculate distances
        matches = [[fuzz.ratio(from_string, to_string) / 100 for to_string in to_list] 
                    for from_string in from_list]
        
        # Get best matches
        mappings = [to_list[index] for index in np.argmax(matches, axis=1)]
        scores = np.max(matches, axis=1)
        
        # Prepare dataframe
        matches = pd.DataFrame({'From': from_list,'To': mappings, 'Similarity': scores})
        return matches
        
custom_model = MyModel()
model = PolyFuzz(custom_model)
'''