#!/usr/bin/env python3

'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

This file extracts the raw category tag from all articles' infoboxes and
separates the articles' body texts from their infobox texts. After, it parses
the infobox texts into a key-value dictionary to render the infoboxes searchable.
It then stores the new [title, full text, hyperlinks, coordinates, raw category,
curated category, infobox dictionary] lists in a database, organized by curated 
category. The file containes code fragments from older category extraction methods
that have since been abandoned.
'''

import sys
from data_processor import *
import re 
import sys
import os

COORDINATES = True
DIRECTORY = "coordinate_articles_new/"
PREPOSITIONS = {"in", "by", "on", "at", "of", "with", "near"}

# Parses the infobox text into many key-value pairs;
# the process is highly deregulated and possibly
# disorganized
def parse_infobox(text, tag, dictionary):
   
    # Return if there is no parsable text
    if text is None:
        return None

    # Split the text box into its constituent entries
    segments = re.split("\n[ ]*\|", text)
    
    for segment in segments:

        # Split each entry into its key-value pair
        index = segment.find("=")

        # Determine if there is a pair or just a single entry
        if index != -1:
            key = segment[:index].strip(" \n\t|").lower().replace("_", " ")
            value = segment[index + 1:].strip(" \n\t|")
            if value == "":
                value = None
        else:
            key = segment.strip(" \n\t").lower().replace("_", " ")
            value = None

        # Skip any empty string key
        if key == "":
            continue

        # Prepend subcategory if this isn't the primary infobox
        if tag is not None:
            key = tag + "_" + key

        # Handle duplicate key names (if they exist)
        if key in dictionary:
            i = 1
            while (key + "_" + str(i)) in dictionary:
                i += 1
            dictionary[key + "_" + str(i)] = value
        else:
            dictionary[key] = value

# Returns the category of the infobox as well as the index of the infobox AFTER the
# category tag
def extract_raw_category(match, text):

    # Can't use regex to extract {{..}} since there is no way of counting the
    # number of recursive {{.}}'s in the infobox
    index = match.end(0)
    category = ""
    # Extract infobox category
    while text[index] != "|" and text[index] != "<" and text[index] != "{" and text[index] != "}":
        category += text[index]
        index += 1
    # Edgecase for geobox
    if category.find("category_hide=yes") != -1:
        index = text.find("|", index) + 1
        category = ""
        while text[index] != "|" and text[index] != "<" and text[index] != "{" and text[index] != "}":
            category += text[index]
            index += 1

    # Trim spaces and newlines from front and back
    category = category.replace("/", " ")
    category = category.replace("_", " ")
    category= category.strip(" \n\t").lower()
    # Remove ' begin' if it exists at the end of category
    if len(category) >= 6:
        if category[-6:] == " begin":
            category = category[:-6]
    # Set category to 'None' if it is ""
    if category == "":
        category = None

    return category, index

# Locates the infobox in the text, retrieves its category tag,
# and returns the text associated with it; returns the raw category
# tag, all raw infobox text, the infobox text to parse, and the 
# text of the body of the article, excluding infoboxes
def extract_infobox(text):

    # Find start of infobox
    match = re.search("{{[iI]nfobox|{{[ \n][iI]nfobox|{{[gG]eobox[ \n]\||{{[ \n][gG]eobox[ \n]\||{{[gG]eobox\|", text)
    
    if match:
    
        # Obtain the category and the index to begin the for the parsable infobox search from
        category, index = extract_raw_category(match, text)
       
        # Find the beginning of the part of the infobox that will be parsed
        start = index
        # Find the start of the part that will be parsed into the infobox dictionary
        if text[index] == "}":
            return category, text[match.start(0):index + 1], None, text[0:match.start(0)] + text[index + 1:]
        # If text[index] is either one of these, we need to find the next "|"
        elif text[index] == "<" or text[index] == "{":
            start = re.search("\n[ ]*\|", text[index:])
            if start is not None:
                start = start.end(0)

        # Find the end of the infobox so that it can be extracted;
        # poorly formatted infoboxes (usually missing a '}}' pair)
        # like 'Acheritou' are excluded
        bracket_count = 0
        while bracket_count >= 0:
            # If the end of the infobox can't be found, don't return it
            if index == len(text):
                return category, None, None, text
            if text[index] == "{":
                bracket_count += 1
            elif text[index] == "}":
                bracket_count -= 1
            index += 1

        # Verify that a parsable infobox was able to be extracted
        if start is None or start > index:
            return category, text[match.start(0):index + 1], None, text[0:match.start(0)] + text[index + 1:]

        # The last field allows for infoboxes to be removed from body text
        return category, text[match.start(0):index + 1], text[start:index - 1], text[0:match.start(0)] + text[index + 1:]
        
    return None, None, None, text

# Removes all ill-formatted categories and classifies their articles
# as "Uncategorized"
def format_raw_category(key):
    if key is None:
        return None
    # Find all unformatted categories and place articles in "Uncategorized"
    if "\n" in key or "\t" in key or "=" in key or "[" in key or ":" in key or ">" in key or "#" in key:
        return None
    if key == "settelement" or key == "settlemen" or key == "settlementعمكم احمد الفضاله":
        return "settlement"
    return key

# Extracts the infobox as well as its category and stores
# all infobox entries in a dictionary; extracts infobox and body
# text as well
def extract_raw_structured_data(article):

    # Initialize variables and perform initial extraction
    body_text = article[1]
    infobox_dict = {}
    raw_category, infobox_text, parsable_text, body_text = extract_infobox(body_text)

    # If infobox text was found, try to find more infoboxes
    if infobox_text is not None:
        parse_infobox(parsable_text, None, infobox_dict)

        # While there are still infoboxes to search for, search and parse
        while True:
            subcategory, text, parsable_text, body_text = extract_infobox(body_text)
            if text is None:
                break
            infobox_text += "\n" + text
            check = True
            parse_infobox(parsable_text, subcategory, infobox_dict)
    
    # Store the infobox text and the body text in the dictionary
    infobox_dict["Infobox Text"] = infobox_text
    match = re.match("\|}|{\|[ \n]*\|}", body_text)
    if match is not None:
        body_text = body_text[match.end(0):]
    infobox_dict["Body Text"] = body_text.strip(" \n\t")

    # Format the raw category if needed
    raw_category = format_raw_category(raw_category)
    
    return raw_category, infobox_dict

# Returns the plural form of the category option passed in
def pluralize(term):
    if term == "person":
        return "people"
    if term[-1] == "y" and term != "valley":
        return term[:-1] + "ies"
    if term[-2:] == "sh" or term[-2:] == "ch":
        return term + "es"
    return term + "s"

# Searches through the raw tag to try to find 'category'
def search_raw_tag(category, category_pl, raw_tag, raw_tag_split):

    # If the raw_tag is 'None' or is designated "former", return
    if raw_tag is None or "former" in raw_tag:
        return None
    searching_nrhp = category == "historic place"

    # Search the raw tag
    if " " not in category:
        for word in raw_tag_split:
            if word == category or word == category_pl or\
                    (word == "nrhp" if searching_nrhp else False):
                return category

    # Simply do an 'in' search for category
    else:
        if category in raw_tag or category_pl in raw_tag or\
                ("nrhp" in raw_tag if searching_nrhp else False):
            return category

# Checks to see if one of the forbidden prepositions proceeds the search term in the hyperlink
def preposition_brake(term, term_pl, hyperlink):

    index = max(hyperlink.find(term), hyperlink.find(term_pl))

    # Verify that no preposition came before this term
    if index != -1:
        for prep in PREPOSITIONS:
            prep_index = hyperlink.find(" " + prep + " ")
            # Return false if the prep came before the term
            if prep_index != -1 and prep_index < index:
                return False
        return True
    return False

# Search through all category hyperlinks for 'category'
def search_category_hyperlinks(category, category_pl, category_hyperlinks, excludables):

    # Iterate through all category hyperlinks
    for hyperlink in category_hyperlinks:
        hyperlink = hyperlink.lower()
        
        # Verfify that the hyperlink doesn't contain an excludable term; we assume that all
        # hyperlinks of an article which we would misclassify contain an excludable term,
        # while not all hyperlinks of a valid article for 'category' do
        skip = False
        for excludable in excludables:
            if excludable.lower() in hyperlink or pluralize(excludable.lower()) in hyperlink:
                skip = True
                break
        if skip:
            # Specifically induce a permanent skip for these categories
            if category == "college" or category == "university":
                return None
            continue

        if " " not in category:
            # Format the hyperlink and remove the first "category:" term if 'category' is one word
            hyperlink_split = re.split("[ -_:,\|]", hyperlink.strip("\n :{}[]|,._-"))[1:]

            # Iterate through all words in the hyperlink
            for i in range(len(hyperlink_split)):

                # If one of the words matches the category option
                if hyperlink_split[i] == category or hyperlink_split[i] == category_pl:
                    return category
                # If the category stipulates "former" or a preposition is found, skip 'hyperlink'
                if hyperlink_split[i] == "former" or hyperlink_split[i] in PREPOSITIONS:
                    break

        else:
            # Verify that no preposition came before this term
            if preposition_brake(category, category_pl, hyperlink):
                return category

    return None

# Wrapper for the two search functions
def search_for_curated_category(category, raw_tag, raw_tag_split, category_hyperlinks, excludables):
    category_pl = pluralize(category)
    label = search_raw_tag(category, category_pl, raw_tag, raw_tag_split)
    if label is not None:
        return label
    return search_category_hyperlinks(category, category_pl, category_hyperlinks, excludables)

# Returns the extracted curated category from the category hyperlinks and
# the extracted raw tag (to make sure that the curated tags contain the
# coverage of well-known entities that the raw tags do)
def extract_curated_category(article, raw_tag, options):
    
    if raw_tag ==  "u.s. county" and "county" in options:
        return "county"
    if raw_tag ==  "lake" and "lake" in options:
        return "lake"
    if raw_tag ==  "mountain" and "mountain" in options:
        return "mountain"
    if (raw_tag == "islands" or raw_tag == "island") and "island" in options:
        return "island"

    # Get the set of category hyperlinks to search through
    category_hyperlinks = get_hyperlinks(article, category=True)
    # Split the raw tag so that we can search through it if 'category' is one word
    raw_tag_split = (None if raw_tag is None else re.split("[ -_:,\|]", raw_tag))

    # Iterate through all category options
    for category in options.keys():
        # Skip nature tags when appropriate; update to move with Curated_Catetegories.txt
        if (category == "lake" or category == "mountain" or category == "forest"
                or category == "desert" or category == "sea" or category == "protected area"
                or category == "island" or category == "valley" or category == "bay" or
                category == "river") and raw_tag == "settlement":
            continue
        label = search_for_curated_category(category, raw_tag, raw_tag_split, category_hyperlinks, 
                options[category][1])
        if label is not None:
            return category
        # Iterate through synonyms
        for synonym in options[category][0]:
            label = search_for_curated_category(synonym, raw_tag, raw_tag_split, category_hyperlinks, 
                    options[category][1])
            if label is not None:
                return category

        # This is the only exception to the synonym search system, since the
        # differece between the raw tag and hyperlink categories will create misclassifications
        if category == "territory":
            label = search_for_curated_category("settlement", raw_tag, raw_tag_split, category_hyperlinks, options[category][1])
            if label is not None:
                return "populated place"
    return None

# Adds the article 'i' to the proper category in both dictionaries
def store_categories(i, raw_categories, curated_categories, raw_category, 
        curated_category, infobox_dict):

    # Process raw category
    if raw_category is not None:

        # Add a new category to the dictionary if needed
        if raw_category not in raw_categories:
            raw_categories[raw_category] = 0
            print("Added raw category: " + raw_category)

        raw_categories[raw_category] += 1
    # If there was a valid raw category extracted
    else:
        raw_categories["Uncategorized"] += 1

    # Process curated category
    if curated_category is not None:

        # Add a new category to the dictionary if needed
        if curated_category not in curated_categories:
            curated_categories[curated_category] = []
            print("Added curated category: " + curated_category)
    
        # Infobox dictionary contains raw infobox text and article text separated from infobox
        curated_categories[curated_category].append([i[0], i[1], i[2], i[3], 
            raw_category, curated_category, infobox_dict])
    else:
        curated_categories["Uncategorized"].append([i[0], i[1], i[2], i[3], 
            raw_category, curated_category, infobox_dict])

# Takes in both dictionaries of categories and saves them; writes
# the summary of the article distribution across categories as well as
# the list of categories to a .txt file
def save_categories(categories, num_total, raw=False):

    # Create full array
    full_array = 0
    if not raw:
        file_name = "Categories.txt"
    else:
        file_name = "Raw_Categories.txt"

    if not os.path.exists(CONST_PREFIX + DIRECTORY):
        os.makedirs(CONST_PREFIX + DIRECTORY)
    with open(CONST_PREFIX + DIRECTORY + file_name, "w") as f:

        # Print and save category data
        for key in sorted(categories.keys()):

            # Print the number of articles in the category
            if not raw:
                # Sort the articles!
                categories[key].sort()
                print("Category: " + key  + "    Number of Articles: " + str(len(categories[key])))
            else:
                print("Category: " + key  + "    Number of Articles: " + str(categories[key]))

            if key != "Uncategorized":
                if not raw:

                    # Verify that all articles in curated categories have the proper category stored
                    for i in categories[key]:
                        i[5] = key

                    full_array += len(categories[key])
                else:
                    full_array += categories[key]

            # Save the list of articles in the curated category
            if not raw:
                # Save, replacing all spaces with underscores for the names
                save_data(categories[key], DIRECTORY , key.replace(" ", "_"))
                # Write the category data to the summary .txt file
                f.write(key + " = " + str(len(categories[key])) + "\n")
            else:
                # Write the category data to the summary .txt file
                f.write(key + " = " + str(categories[key]) + "\n")

    # Print aggregate data
    print("Number of categories: " + str(len(categories)))
    if not raw:
        print("Obtained curated categories for " + str(full_array) + " of " + str(num_total) + " articles!")
    else:
        print("Obtained raw categories for " + str(full_array) + " of " + str(num_total) + " articles!")

    print("Finished processing and saving all arrays!")

# Build the dictionary of synonymous and excludable terms for each category option
def build_category_options(category_list):

    categories = {}
    
    # Parse the list so that synonyms and excludables are held separately
    for i in category_list:
        segments = i.split(": ")
        if len(segments) == 1:
            categories[segments[0]] = [[], []]
            continue
        synonyms = (segments[1].split(", ") if segments[1] != "" else [])
        if len(segments) == 2:
            categories[segments[0]] = [synonyms, []]
            continue
        excludables = segments[2].split(", ")
        categories[segments[0]] = [synonyms, excludables]
    
    # Print the synonym-excludable dictionary
    for i in categories.keys():
        print("Curated Category: " + i)
        print("\t\t\t\tSynonyms:")
        for j in categories[i][0]:
            print("\t\t\t\t\t" + j)
        print("\t\t\t\tExcludables:")
        for j in categories[i][1]:
            print("\t\t\t\t\t" + j)

    return categories

# Iterate through all articles and extract all structured data possible
# from them, including category and infobox data; store by category;
# [title, full text, hyperlinks, coordinates, raw category, curated category, infobox dictionary]
if __name__ == '__main__':

    # Load the list of curated category options into a dictionary
    curated_category_options = build_category_options(load_category_list())

    # Declare dictionary of arrays for categories; articles will be stored by
    # their curated category; the raw_category dictionary is used to track the
    # distribution of raw categories
    raw_categories = {}
    raw_categories["Uncategorized"] = 0

    curated_categories = {}
    curated_categories["Uncategorized"] = []

    # Read in list of all off-world titles that should have their coordinates removed
    invalid_titles = set(load_data("miscellanious_files/", "off_world_titles_from_raw_categories", verbose=True))

    # Load array of articles to categorize and extract info boxes from
    if not COORDINATES:
        array = load_uncategorized_alphabetized_article_array("full", verbose=True)
    else:
        array = load_uncategorized_alphabetized_coordinate_array("full", verbose=True)

    # Extract all articles
    for i in array:

        # Remove coordinates from invalid articles (or skip, depending on what array was initially loaded)
        if i[0] in invalid_titles:
            if COORDINATES:
                continue
            i = (i[0], i[1], i[2], None)

        # Get raw category
        raw_category, infobox_dict = extract_raw_structured_data(i)
        print(i[0] + " Raw:     " + str(raw_category))
        # Get raw category
        curated_category = extract_curated_category(i, raw_category, curated_category_options)
        print(i[0] + " Curated:     " + str(curated_category))
        
        # Save categories into dictionaries
        store_categories(i, raw_categories, curated_categories, raw_category, 
                curated_category, infobox_dict)

    # Format and save category dictionaries
    save_categories(raw_categories, len(array), raw=True)
    save_categories(curated_categories, len(array))


