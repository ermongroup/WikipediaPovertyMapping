#!/usr/bin/env python3
'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

Loads the xml articles and extracts the text and hyperlinks from each one. 
A new array is built of these articles. All articles that are also contained 
in the coordinate element array are also included here as well, and they also
retain their coordinate tag as the element at index 3, as (title, text, hyperlink, coordinate)
tuples. All other articles are stored as (title, text, hyperlink, None) 
tuples.
'''

import sys
from data_processor import *
from sequester_coordinate_articles import extract_text
from sequester_coordinate_articles import extract_hyperlinks

# Process the article according to the outlined goals
def process_articles(key, full_array, coordinate_element_array):

    array = load_xml_array(key, verbose=True)
    processed = []

    for article in array:
        coordinate_element = get_article(article[0], loaded=coordinate_element_array, safeguard=False)
        # If the article didn't contain coordinates, process it
        if coordinate_element is None:
            # These functions are written in sequester_coordinate_articles.py
            text = extract_text(article[1])
            hyperlinks = extract_hyperlinks(text)
            processed.append((article[0], text, hyperlinks, None))
        else:
            processed.append(coordinate_element)
    
    processed.sort()
    save_data(processed, "articles/", key, verbose=True)

    full_array += processed
    print("Added " + CONST_PREFIX + "articles/" + key + ".txt")

# Processes each array of xml articles and saves a combined version
# of the newly-created arrays
if __name__ == '__main__':
   
    # Declare array for all articles
    full_array = []
    # Load array for checking whether article has coordinates
    coordinate_element_array = load_coordinate_element_array("full", verbose=True) 

    for letter in ALPHABET:
        process_articles(letter, full_array, coordinate_element_array)

    process_articles("Special", full_array, coordinate_element_array)
    process_articles("File", full_array, coordinate_element_array)
    process_articles("Category", full_array, coordinate_element_array)
    process_articles("Template", full_array, coordinate_element_array)
    
    full_array.sort()
    print("Sorted " + CONST_PREFIX + "articles/Full.txt")

    save_data(full_array, "articles/", "Full", verbose=True)
    print("Finished processing and saving all arrays!")
