#!/usr/bin/env python3
'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

This file parses the large xml file that all wikipedia articles are stored
in into .txt files storing all articles beginning with a certain letter,
all File: articles, all Template: articles, and all Category: articles. 

OLDER NOTE:
Because the xml file is so large, .txt files in each of these categories 
(letters, keywords) are stored in 20000 article blocks so that the parser 
does not crash. This distributed storage system is then rectified in 
the data_processing.py file.

************TODO***********:
    Update directory paths, and change storage method so that it
    takes advantage of atlas RAM size.
'''

import sys
sys.path.append("/atlas/u/esheehan/wikipedia_project/dataset/text_dataset/dataset_modules")
from data_processor import CONST_PREFIX
from data_processor import ALPHABET
import numpy as np
import os

# String constants for title manipulation
FILE = "File:"
CATEGORY = "Category:"
TEMPLATE = "Template:"

# Number of articles that can be in array before it is loaded off RAM and refreshed
CONST_MAX_ARTICLES_PER_ARRAY = 20000

# Verifies if one of the arrays is approaching the max value and, if so, saves it and refreshes
# its data in the array dictionary
def array_check(val, current_arrays, terminate=False):
    
    num, array = current_arrays[val]
    # Array needs to be offloaded
    if (len(array) == CONST_MAX_ARTICLES_PER_ARRAY) or terminate:
        array.sort()
        directory = CONST_PREFIX + "xml_articles/xml_unsorted/" + val + "/" 
        if not os.path.exists(directory):
            os.makedirs(directory)
        articles = directory + val + "_" + str(num) + ".txt"
        f = open(articles, "w")
        f.write(str(array))
        f.close()

        # Offload RAM
        current_arrays[val] = (num + 1, [])
        print("Offloaded and Saved " + articles)

# Returns the text from an html page/article along with identifying booleans
def build_page(namespaces, meta, external_line):

    # Initialize external parameters
    add_article, add_file, add_category, add_template = False, False, False, False
    title = ""
    # Create string representation of article
    article = external_line
    # Process page internals
    for internal_line in meta:

        # Build article string
        article += internal_line
        # If page has ended, break to outer loop
        if "</page>" in internal_line:
            break

        # Process title line if no line with <title> has been seen before
        if "<title>" in internal_line and title == "":

            # Store title line between the <title> and </title> substrings
            title = internal_line[(internal_line.find("<title>") + 7):(internal_line.find("</title>"))]
            # If the title contains a forbidden namespace, disregard article
            for namespace in namespaces:
                if namespace in internal_line:
                    return title, article, False, False, False, False

            # Determine where the article should be added
            if FILE == internal_line[(internal_line.find("<title>") + 7):(internal_line.find("<title>") + 12)]:
                add_file = True
            elif CATEGORY == internal_line[(internal_line.find("<title>") + 7):(internal_line.find("<title>") + 16)]:
                add_category = True
            elif TEMPLATE == internal_line[(internal_line.find("<title>") + 7):(internal_line.find("<title>") + 16)]:
                add_template = True
            else:
                add_article = True

        # If it is a redirect page, skip it
        if "<redirect title=" in internal_line:
            return title, article, False, False, False, False
    return title, article, add_article, add_file, add_category, add_template

# Will store tuples of (title, article/page) in many different arrays and
# save them to .txt files
if __name__ == '__main__':

    # Initialize article array storage system; (array number, list)
    current_arrays = {}
    for letter in ALPHABET:
        current_arrays[letter] = (1, [])
    current_arrays["Special"] = (1, [])
    current_arrays[FILE[:-1]] = (1, [])
    current_arrays[CATEGORY[:-1]] = (1, [])
    current_arrays[TEMPLATE[:-1]] = (1, [])

    # Build namespaces to exclude
    namespaces = set()
    with open(CONST_PREFIX + "xml_articles/excludable_namespaces.txt") as f:
        for namespace in f:
            namespaces.add(namespace[:-1] + ":")

    # Process files
    with open(CONST_PREFIX + "xml_articles/enwiki-20180620-pages-meta-current.txt", "r") as meta:
        
        count = 0
        added1, added2, added3, added4 = 0, 0, 0, 0
        for external_line in meta:

            # Process page
            if "<page>" in external_line:

                # Build page
                title, article, add_article, add_file, add_category, add_template = build_page(namespaces, meta, external_line)

                # Check if page should be added to an article array, and add it accordingly
                val = ""
                if add_file:
                    val = FILE[:-1]
                    added1 += 1
                elif add_category:
                    val = CATEGORY[:-1]
                    added2 += 1
                elif add_template:
                    val = TEMPLATE[:-1]
                    added3 += 1
                elif add_article:
                    val = title[0].upper()
                    if not val in current_arrays:
                        val = "Special"
                    added4 += 1

                # Verify its storage status and update it if necessary
                if not val == "":
                    current_arrays[val][1].append((title, article))
                    array_check(val, current_arrays)

                count += 1
                if count % 10000 == 0:
                    print("Total Seen: " + str(count) + "    Articles: " + str(added4) + "    Files: " 
                            + str(added1) + "    Categories: " + str(added2) + "    Templates: " + str(added3))

    # Save all arrays
    for key in current_arrays:
        array_check(key, current_arrays, terminate=True)
    print("SAVED ALL!")

