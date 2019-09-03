# WikipediaPovertyMapping
This repository contains the implementation of our _KDD2019_ paper: [__Predicting Economic Development Using Geolocated Wikipedia Articles__](https://dl.acm.org/citation.cfm?doid=3292500.3330784).

<p align="center">
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/global_wiki.png" width="500"/>
</p>

Here, we present the code for parsing a Wikipedia xml dump into its constituent geolocated articles,
converting them to human readable format, training a Doc2Vec model over them, and building and training
a model using the Doc2Vec embeddings to predict poverty via (1) the 10 closest articles and (2) the articles
as well as Nightlights imagery.


We will be updating and cleaning the code for distribution and can field concerns or requests at esheehan@stanford.edu or chenlin@stanford.edu

# Predicting Economic Development Using Geolocated Wikipedia Articles

## Downloading Geolocated Articles
will be updated soon!

## Parsing Geolocated Articles
will be updated soon!

## Training Doc2Vec on Geolocated Articles
will be updated soon!

## Socioeconomic Data
Our ground truth dataset consists of data on asset ownership from the Demographic and Health Surveys [DHS](https://www.dhsprogram.com/) and [Afrobarometer](http://http://afrobarometer.org). DHS is a set of nationally representative household surveys conducted periodically in
most developing countries, spanning 31 African countries. Afrobarometer (Round 6) is a survey across 36 countries in
Africa, assessing various aspects of welfare, education, and infrastructure quality.

## Nightlight Imagery Dataset
We use high-resolution night-time imagery from VIIRS [Elvidge et al., 2017]. Each of the imagery is of shape (255, 255). For each point in the AWI data that we train on, we obtain a nightlights image centered on its coordinate. The size of these
images was set to (5 km × 5 km) due to the (maximum) 5km noise that each location has. 

## Models
We provide the code for the models in
```
models/
```
### Wikipedia Embedding Model
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Wikipedia_Embedding_Model.png" width="500"/>

### Wikipedia Nightlight Multi-modal Model
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Multi-modal_Architecture.png" width="500"/>
To train the multi-modal model, run
```
python 
```

If you use this repository, please cite our paper:

	Sheehan, E., Meng, C., Tan, M., Uzkent, B., Jean, N., Burke, M., Lobell, D. and Ermon, S., 2019, July. Predicting Economic Development using Geolocated Wikipedia Articles. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2698-2706). ACM.

