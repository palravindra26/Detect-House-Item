# Detect House Item

Predicts common household items that are present in the image.
Inspired by [Amenity Detection and Beyond — New Frontiers of Computer Vision at Airbnb](https://medium.com/airbnb-engineering/amenity-detection-and-beyond-new-frontiers-of-computer-vision-at-airbnb-144a4441b72e) and [Daniel Bourke](https://www.mrdbourke.com/42days)

Dataset Used : [Open Images](https://opensource.google/projects/open-images-dataset)

Model Used : [Retinanet(keras-retinanet backend - resnet50)](https://github.com/fizyr/keras-retinanet)

Live App : [DetectHouseItem](https://detecthouseitem.herokuapp.com/)

Note: This app is deployed on heroku's free dyno which only provide 500 MB of RAM. The total size of the app is around 430 MB (+ 140 MB trained model) 
which make the site's response time slow and  to crash sometime.

Web Framework used : [streamlit](https://www.streamlit.io/)

Sample Image:

![sample image](https://github.com/rishabhvarshney14/Detect-House-Item/blob/master/sample.png)

## About Files

- app : This is the main directory which contains all the required files.
- scripts : Thie the directory which ontains python files used during training.
    - downloadOI : This is used to download images from Open Images Dataset and form proper annotation file as required by keras-retinanet.
- notebooks : Jupyter Notebooks that I used to to train the model.

## How to Use

To train the model first download the images from Open Images using scripts/downloadOI.py it will create a folder 'data' which contains three folders
1. images : this will contains all the images downloaded.
2. classes : this contains classes.csv file which is required by keras-retinanet.
3. annotation : this contains annotation files (train.csv, test.csv, validation.csv) which is required by keras-retinanet.

then used notebooks/training.ipynb file to train the model.
