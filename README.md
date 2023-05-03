# aml-hft-attention
 The solution is designed to predict smooth average trend for XBTUSD trading. The structure of the Readme is as follows:

 1. Overview of functionality
 2. Requirements
 3. Manual on usage
 4. Docker file structure
 5. Description of model applied

 # 1. Overview of functionality

 The docker contains pre-trained model on prediction of smooth average trend for XBTUSD for 700 ticks ahead. The model accepts snapshots of XBTUSD orderbook and outputs three-value signal on direction of predicted mid-price. Docker allows to train the model based on data feeded for predictions. 

# 2. Requirements

In order to use the docker, please install the docker engine as described in the link: https://docs.docker.com/engine/install/.

# 3. Manual on usage

Note that any docker commands might need to be launched with administrative privilages (etc., sudo).
In order to create docker image, first build the docker image:

    cd ./aml-hft-attention
    docker build xbtusd -t trade ./

Then run the image with the attached volume 'shared' and 9999 port forwarding to the port suitable for usage:

    docker run -p 9999:[YOUR_PORT] -v ./docker/shared:/home/shared xbtusd

This will create a tcp server in selected port which can be interacted with via terminal. For instance, you can use 'nc' app for linux:

    nc localhost [YOUR_PORT]

In order to receive list of commands available, input 'help' to the terminal, that will give list of commangs and description.
To send data for prediction, send the orderbook snapshots to ./aml-hft-attention/docker/shared/inp_data/ in csv format. For example, you can use ./aml-hft-attention/dataset/test/.
Then, enter the command:

    predict [FILE_NAME].csv

The resulting file will be placed in ./aml-hft-attention/docker/shared/out_data/. WARNING: out files from data with the same names will be overwritten. 

After a prediction made, the docker stores processed data in ./aml-hft-attention/docker/shared/past_data.csv. This data will be extended after each prediction and can be used for training. To do this, send the 'train' command along with number of epochs to train (the default value is 5):

    train [NUMBER_OF_EPOCHS]

In order to delete accumulated data for training, please, use 'clear' command.

# 4. Docker file structure

Docker utilize ~7.1 Gb of free space. Along with it there is shared volume attached for input / output data as well as pre-trained model. In order to change the shared volume location, copy the ./aml-hft-attention/docker/shared/ folder to desired location and mention that path during docker run command:

    docker run -p 9999:[YOUR_PORT] -v [PATH_TO_VOLUME]:/home/shared xbtusd

The model file is a binary archive representing a python dict with the following data:

    'model' :   a pytorch pre-train model. Note that in order to import external model into the docker,you have to add the model's schema to the ./aml-hft-attention/src/models.py
    'pipe' : fitted pipeline for data transdormation
    'scores' : a dict with evaluation metrics gained during the training
    'params' : a dict with parametrs used for training

In order to unpack the model archive, please, use either torch loader or the pickle library.

# 5. Description of model applied

The model applied utilizes algorithm consisting of double attention stages and Gated Recurrent network in combination with ML feature extraction. Model outputs three signals - stationary, Down and Up. Input data consist of reconstructed LOB.
