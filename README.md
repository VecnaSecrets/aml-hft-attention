# aml-hft-attention
 The solution is designed to predict smooth average trend for XBTUSD trading. The structure of the Readme is as follows:

 1. Overview of functionality
 2. Requirements
 2. Manual on usage
 3. Description of model applied

 # 1. Overview of functionality

 The docker contains pre-trained model on prediction of smooth average trend for XBTUSD for 700 ticks ahead. The model accepts snapshots of XBTUSD orderbook and outputs three-value signal on direction of predicted mid-price. Docker allows to train the model based on data feeded for predictions. 

# 2. Requirements

# 3. Manual on usage

Note that any docker commands might need to be launched with administrative privilages (etc., sudo).
In order to create docker image, first build the docker image:

    cd ./aml-hft-attention
    docker build xbtusd -t trade ./

Then run the image with the attached volume 'shared' and 9999 port forwarding to the port suitable for usage:

    docker run -p 9999:[YOUR_PORT] -v ./docker/shared:/home/shared xbtusd

This will create a tcp server in selected port which can be interacted with via terminal. For instance, you can use 'nc' app for linux:

    nc localhost YOUR_PORT

In order to receive list of commands available, input 'help' to the terminal, that will give list of commangs and description.
To send data for prediction, send the orderbook snapshots to ./aml-hft-attention/docker/shared/inp_data/ in csv format. For example, you can use ./aml-hft-attention/dataset/test/.
Then, enter the command:

    predict FILE_NAME.csv

The resulting file will be placed in ./aml-hft-attention/docker/shared/out_data/. WARNING: out files from data with the same names will be overwritten. 

After a prediction made, the docker stores processed data in ./aml-hft-attention/docker/shared/past_data.csv. This data will be extended after each prediction and can be used for training. To do this, send the 'train' command along with number of epochs to train (the default value is 5):

    train NUMBER_OF_EPOCHS