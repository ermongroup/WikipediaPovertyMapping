# WikipediaPovertyMapping
Implementation of Geolocated Articles Processing and Poverty Mapping

Here, we present the code for parsing a Wikipedia xml dump into its constituent geolocated articles,
converting them to human readable format, training a Doc2Vec model over them, and building and training
a model using the Doc2Vec embeddings to predict poverty via (1) the 10 closest articles and (2) the articles
as well as Nightlights imagery.

We will be updating and cleaning the code for distribution and can field concerns or requests at esheehan@stanford.edu.

# Predicting Economic Development Using Geolocated Wikipedia Articles

This repository contains the implementation of our _KDD2019_ paper on [__Predicting Economic Development Using Geolocated Wikipedia Articles__](http://delivery.acm.org/10.1145/3340000/3330784/p2698-sheehan.pdf?ip=171.64.70.130&id=3330784&acc=OPEN&key=AA86BE8B6928DDC7%2E0AF80552DEC4BA76%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1564771483_7a50df95ffb334f1023ed223dd8fd3b1).

You can find our paper [here](http://delivery.acm.org/10.1145/3340000/3330784/p2698-sheehan.pdf?ip=171.64.70.130&id=3330784&acc=OPEN&key=AA86BE8B6928DDC7%2E0AF80552DEC4BA76%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1564771483_7a50df95ffb334f1023ed223dd8fd3b1).

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
will be updated soon!
### Wikipedia Embedding Model
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Wikipedia_Embedding_Model.png" width="500"/>

### Multi-modal Model
<img src="https://github.com/ermongroup/WikipediaPovertyMapping/blob/master/images/Multi-modal_Architecture.png" width="500"/>


If you use this repository, please cite our paper:

	Sheehan, E., Meng, C., Tan, M., Uzkent, B., Jean, N., Burke, M., Lobell, D. and Ermon, S., 2019, July. Predicting Economic Development using Geolocated Wikipedia Articles. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2698-2706). ACM.

