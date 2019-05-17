#!/usr/bin/env python3
'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

Loads the arrays of xml articles, extracts each one that contains
a geolocation tag (in the form of coordinates), and builds a new array
of these articles structured as (title, body text, hyperlinks, coordinates)
tuples; the body text of the article is also cleaned up by replacing
xml code abbreviations with their human-readable counterparts; saves
the arrays in the 'coordinate_articles' directory.
'''

import sys
sys.path.append("/atlas/u/esheehan/wikipedia_project/dataset/text_dataset/dataset_modules")
from data_processor import *
import re

CONST_VALID_COORDINATE_CHARS = "0123456789.-"
CONST_VALID_COORDINATE_DIRECTIONS = "NSEW"

# Extracts all hyperlinks from the text body and stores them in a python list
def extract_hyperlinks(text):

    hyperlinks = set()
    index = 0

    # Iterate through whole article
    while index < len(text):

        # If a letter is '['
        if text[index] == "[":
            # If the letter following it is also '[', then it is a hyperlink
            if index + 1 < len(text) and text[index + 1] == "[":

                # Build hyperlink
                index += 2
                link = ""
                while index < len(text) and not text[index] == "]" and not text[index] == "|":
                    link += text[index]
                    index += 1
                hyperlinks.add(link)

        index += 1

    # Sort to make searchable via binary search
    hyperlinks = list(hyperlinks)
    hyperlinks.sort()

    return hyperlinks

# Removes the text from the article body and changes any xml
# syntax to its human-readable corollary
def extract_text(article):
    
    start = article.find("<text xml:space=\"preserve\">") + len("<text xml:space=\"preserve\">")
    end = article.find("</text>", start)
    text = article[start:end]

    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("&apos;", "\'")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")

    return text

# Returns whether the given string is comprised of all characters which
# are valid for numerical coordinates
def validate_coordinate_string(string):
    for char in string:
        if not char in CONST_VALID_COORDINATE_CHARS:
            return False
    if string == "" or string == "." or string == "-" or ".." in string or "--" in string:
        return False

    return True

# Transforms a numerical string to either an int or a float
def to_number(string):
    if "." in string:
        return float(string)
    return int(string)

# Parses the string coordinate into a numerical representation;
# works on one degree-minute-second-direction sequence;
# assumes that the string begins with the degree numbers and not
# '|'
def parse_coordinates(string):

    # Parse the primary data point, either degrees or latitude
    prev = 0
    index = string.find("|")
    # Iterate through "|...|" segments to find the start of the coordinate sequence
    while not validate_coordinate_string(string[prev:index]):
        prev = index + 1
        if prev >= len(string):
            return None, None, None, None, None
        index = string.find("|", prev)
        if index == -1:
            return None, None, None, None, None

    data1 = to_number(string[prev:index])
    # If it contains no minute or second data (won't be tripped by lat-long format)
    if string[index + 1].upper() in CONST_VALID_COORDINATE_DIRECTIONS:
        return data1, 0, 0, string[index + 1], index + 3

    # Parse the secondary data point, either minutes or longitude;
    # the second data point is the one that must be tested for lat-long format
    index2 = string.find("|", index + 1)

    if not validate_coordinate_string(string[index + 1:index2]):
        return None, None, None, None, None
    data2 = to_number(string[index + 1:index2])

    # If no second data is contained (must check for "scale", etc. edge case)
    if string[index2 + 1].upper() in CONST_VALID_COORDINATE_DIRECTIONS:
        # Check to see if "scale" tag follows
        if len(string[index2 + 1:]) >= 4:
            if string[index2 + 1: index2 + 5].lower() == "name":
                return data1, data2, None, None, -1

        if len(string[index2 + 1:]) >= 5:
            if string[index2 + 1: index2 + 6].lower() == "scale":
                return data1, data2, None, None, -1

        if len(string[index2 + 1:]) >= 6:
            if string[index2 + 1: index2 + 7].lower() == "source":
                return data1, data2, None, None, -1

        return data1, data2, 0, string[index2 + 1], index2 + 3

    # If the coordinates are in lat-long format (i.e. a non-NSEW AND non-numeric tag follows 2 numbers)
    elif string[index2 + 1] not in CONST_VALID_COORDINATE_CHARS:
        return data1, data2, None, None, -1

    # Parse the tertiary data point (if the format is not lat-long),
    # which must be seconds 
    index3 = string.find("|", index2 + 1)

    if not validate_coordinate_string(string[index2 + 1:index3]):
        return None, None, None, None, None
    seconds = to_number(string[index2 + 1:index3])
    if string[index3 + 1] not in CONST_VALID_COORDINATE_DIRECTIONS:
        return None, None, None, None, None
    return data1, data2, seconds, string[index3 + 1], index3 + 3

# Extracts the coordinates related to the article; if no
# coordinates are available, returns 'None'; coordinate
# tuple is (degrees(°), minutes('), seconds("), Direction(NSEW), 
# degrees(°), minutes('), seconds("), Direction(NSEW)), and thus
# 8-dimensional, if it is in degree format, or (lat, long), and
# thus 2-dimensional, if it is in lat-long format
def extract_coordinates(article):

    # Search for "{{[cC]oord|" in the article
    for word in re.finditer("{{[cC]oord\|", article):

        index = word.end()
        brace_count = 0

        # Track the number of open braces to find the end of the coordinate string
        while brace_count >= 0 and index < len(article):
            if article[index] == "{":
                brace_count += 1
            elif article[index] == "}":
                brace_count -= 1

            index += 1

        if index > len(article):
            continue

        # Element at index - 1 will ALWAYS be a "}"
        coordinate_string = article[word.end():index - 1]
        # Replace any newlines, spaces, etc, that may appear
        coordinate_string = coordinate_string.replace("\n", "")
        coordinate_string = coordinate_string.replace(" ", "")
        coordinate_string = coordinate_string.replace("\t", "")
        coordinate_string = coordinate_string.replace("|||", "|")
        coordinate_string = coordinate_string.replace("||", "|")

        # Search the string for coordinates
        if "display=" in coordinate_string and "title" in coordinate_string:
            data1, data2, seconds1, direction1, end = parse_coordinates(coordinate_string)
            # If something went wrong in the parsing, go to next tag; usually occurs because
            # the article had coordinates at one time, but they were removed, leaving only
            # the tags behind, or there is another, valid coordinate tag in the article
            if data1 is None:
                continue
            # If the coordinates were structured as (lat, long)
            if end == -1:
                return (data1, data2)
            degrees2, minutes2, seconds2, direction2, __ = parse_coordinates(coordinate_string[end:])
            # If something went wrong in the parsing, go to next coordinate tag
            if degrees2 is None:
                continue
            return(data1, data2, seconds1, direction1, degrees2, minutes2, seconds2, direction2)
    
    return None

# Process all the articles in the 'key' array; the articles are already in
# sorted order
def process_articles(key, full_array):

    signs = {"N":1, "S":-1, "E":1, "W":-1}
    # Load xml array
    array = load_xml_array(key, verbose=True)
    # Initialize new array
    coordinate_array = []

    # Read in list of all off-world titles that should have their coordinates removed
    titles = set(load_data("miscellanious_files/", "off_world_titles_from_raw_categories.txt", verbose=True))

    # Iterate over all articles in the array
    for initial_article in array:

        # Remove coordinates from invalid articles 
        if initial_article[0] in titles:
            continue

        # Extract the coordinates from the article, verifying they exist
        coordinates = extract_coordinates(initial_article[1])

        # If the article is geolocated, process and store it
        if coordinates is not None:
            
            # Extract the article's text
            text = extract_text(initial_article[1])
            # Extract the article's hyperlinks
            hyperlinks = extract_hyperlinks(text)

            # Add article to new array after converting coordinates to lat, long
            print(initial_article[0] + "        " + str(coordinates))
            if initial_article[0] == "Nógrádmarcal":
                coordinates = (48.0275, 19.385556)
            if initial_article[0] == "Snake Island (Tasmania)":
                coordinates = (-43.172778, 147.292778)
            if initial_article[0] == "Swan Creek Bridge":
                coordinates = (36, 42, 2, 'N', 93, 5, 8, 'W')
            if len(coordinates) > 2:
                latitude = (coordinates[0] + coordinates[1] / 60.0 + coordinates[2] / 3600.00) * signs[coordinates[3].upper()]
                longitude = (coordinates[4] + coordinates[5] / 60.0 + coordinates[6] / 3600.00) * signs[coordinates[7].upper()]
                coordinates = (latitude, longitude)

            if not(abs(coordinates[0]) > 90 or abs(coordinates[1]) > 180):
                coordinate_array.append((initial_article[0], text, hyperlinks, coordinates))
 
    # Save new array (some arrays may be empty, but, for consistency's sake, we will
    # nonetheless create .txt files for them)
    coordinate_array.sort() # As a sanity check/failsafe
    print("Extracted " + str(len(coordinate_array)) + " geolocated articles...")

    save_data(coordinate_array, "coordinate_articles_fixed/", key, verbose=True)

    full_array += coordinate_array
    print("Added " + CONST_PREFIX + "coordinate_articles_fixed/" + key + ".txt")

# Processes each array of xml articles and saves a combined version
# of the newly-created arrays
if __name__ == '__main__':
   
    # Declare array for all articles
    full_array = []

    for letter in ALPHABET:
        process_articles(letter, full_array)
    
    # Excluding Template, etc., SO BE CAREFUL LOADING ARRAYS
    process_articles("Special", full_array)
    
    print("Extracted " + str(len(full_array)) + " geolocated articles in total...")
    full_array.sort()
    print("Sorted " + CONST_PREFIX + "coordinate_articles_fixed/Full.txt")

    save_data(full_array, "coordinate_articles_fixed/", "Full", verbose=True)
    print("Finished processing and saving all arrays!")

