# WikipediaPovertyMapping
This repository contains the implementation of our _KDD2019_ paper: [__Predicting Economic Development Using Geolocated Wikipedia Articles__](https://dl.acm.org/citation.cfm?doid=3292500.3330784).

<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/global_wiki.png" width="700"/>
</p>

Here, we present the code for parsing a Wikipedia xml dump into its constituent geolocated articles,
converting them to human readable format, training a Doc2Vec model over them, and building and training
a model using the Doc2Vec embeddings to predict poverty via (1) the 10 closest articles and (2) the articles
as well as Nightlights imagery.


We will be updating and cleaning the code for distribution and can field concerns or requests at esheehan@stanford.edu or chenlin@stanford.edu.

# Predicting Economic Development Using Geolocated Wikipedia Articles

## Downloading and Parsing Geolocated Articles
The files for downloading and parsing geolocated articles are located in 

```
article_processing_modules/
```
Here is a discription of what each file does:<br />
**convert_xml_articles.py**:
Loads the xml articles and extracts the text and hyperlinks from each one. 
A new array is built of these articles. All articles that are also contained 
in the coordinate element array are also included here as well, and they also
retain their coordinate tag as the element at index 3, as (title, text, hyperlink, coordinate)
tuples. All other articles are stored as (title, text, hyperlink, None) 
tuples.

**parse_article_data.py**:
This file extracts the raw category tag from all articles' infoboxes and
separates the articles' body texts from their infobox texts. After, it parses
the infobox texts into a key-value dictionary to render the infoboxes searchable.
It then stores the new [title, full text, hyperlinks, coordinates, raw category,
curated category, infobox dictionary] lists in a database, organized by curated 
category. The file containes code fragments from older category extraction methods
that have since been abandoned.

**parse_pages_meta.py**:
This file parses the large xml file that all wikipedia articles are stored
in into .txt files storing all articles beginning with a certain letter,
all File: articles, all Template: articles, and all Category: articles.

**sequester_coordinate_articles.py**:
Loads the arrays of xml articles, extracts each one that contains
a geolocation tag (in the form of coordinates), and builds a new array
of these articles structured as (title, body text, hyperlinks, coordinates)
tuples; the body text of the article is also cleaned up by replacing
xml code abbreviations with their human-readable counterparts; saves
the arrays in the 'coordinate_articles' directory.

<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/geolocated_wikipedia_article.png" width="600"/>
</p>

## Training Doc2Vec on Geolocated Articles
We use [gensim](https://radimrehurek.com/gensim/models/doc2vec.html) doc2vec packege for training the Doc2Vec model. To train the Doc2Vec model on geolocated articles, run:

```
python Doc2Vec/Doc2Vec_modules/train_doc2vec.py
```

## Socioeconomic Data
Our ground truth dataset consists of data on asset ownership from the Demographic and Health Surveys [DHS](https://www.dhsprogram.com/) and [Afrobarometer](http://http://afrobarometer.org). DHS is a set of nationally representative household surveys conducted periodically in
most developing countries, spanning 31 African countries. Afrobarometer (Round 6) is a survey across 36 countries in
Africa, assessing various aspects of welfare, education, and infrastructure quality.

<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/wealth-groundtruth-africa.png" width="260"/>
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/error_range.png" width="260"/>
</p>

## Nightlight Imagery Dataset
We use high-resolution night-time imagery from VIIRS [Elvidge et al., 2017]. Each of the imagery is of shape (255, 255). For each point in the AWI data that we train on, we obtain a nightlights image centered on its coordinate. The size of these
images was set to (5 km × 5 km) due to the (maximum) 5km noise that each location has. 

## Models
We discuss the model architecture in detail in section 3 of the paper.
We provide the code for Nightlight Only Model, Wikipedia Embedding Model and Multi-modal Model in
```
models/
```
### Nightlight Only Model
To train the Wikipedia embedding model, run

```
python models/NL_onlyModel.py
```

### Wikipedia Embedding Model
See section 3.2 for more detail.
<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Wikipedia_Embedding_Model.png" width="500"/>
</p>
	
To train the Wikipedia embedding model, run

```
python models/comboModel.py
```
The model will evaluate the results on Ghana, Malawi, Nigeria, Tanzania, and Uganda.
### Wikipedia Nightlight Multi-modal Model
See section 3.3 for more detail.
<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Multi-modal_Architecture.png" width="500"/>
</p>

To train the multi-modal model, run

```
python models/multi_Doc2VecModel.py
```
The model will evaluate the results on Ghana, Malawi, Nigeria, Tanzania, and Uganda.

## Results
When paired with nightlights satellite imagery,
our method outperforms all previously published benchmarks for
this prediction task, indicating the potential of Wikipedia to inform
both research in the social sciences and future policy decisions.

<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/table1.png" width="1000"/>
</p>

Here is an example:
On the left is the ground truth values of 1074 data points averaged on Admin 2 Level for Tanzania. On the right is the Wikipedia Embedding Model predicted values averaged on Admin 2 level for a model trained on Ghana and tested on Tanzania (r2 = 0.52). The resulting predictions show that the model is capable of generalizing across national boundaries.
<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/GHA_TZA(groundtruth,predicted).png" width="600"/>
</p>

If you use this repository, please cite our paper:
```
@inproceedings{Sheehan:2019:PED:3292500.3330784,
 author = {Sheehan, Evan and Meng, Chenlin and Tan, Matthew and Uzkent, Burak and Jean, Neal and Burke, Marshall and Lobell, David and Ermon, Stefano},
 title = {Predicting Economic Development Using Geolocated Wikipedia Articles},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '19},
 year = {2019},
 isbn = {978-1-4503-6201-6},
 location = {Anchorage, AK, USA},
 pages = {2698--2706},
 numpages = {9},
 url = {http://doi.acm.org/10.1145/3292500.3330784},
 doi = {10.1145/3292500.3330784},
 acmid = {3330784},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {computational sustainability, deep learning, remote sensing},
} 
```
