import os
import socket
from datetime import datetime
from model import Predictor
import pandas as pd
from src.pipeline import snap2feat
import torch as t
import numpy as np

class Server():
    def __init__(self, port, params):
        globals().update(params)
        self.port = port
        self.predictor = Predictor(PATH_MODEL, PATH_TEST_DATA, MAX_TEST_SAMPLES)
        self.server = None
        self.client = None
        if os.path.isfile(PATH_OLD_DATA):
            self.untrained_data = pd.read_csv(PATH_OLD_DATA)
        else:
            self.untrained_data = pd.DataFrame([])
        self.device = 'cude' if t.cuda.is_available() else 'cpu'

        self.commands = {
            'exit' : self.exit,
            'predict' : self.predict,
            'train' : self.train,
            'help' : self.help
        }
        self.description = {
            'exit' : "close the connection",
            'predict' : "[file_name] predict the input data. Data have to stored in ./docker/shared/ind_data/",
            'train' : "[epochs] train data based on earlier predictions. The data for training stored in ./docker/shared/past_data.csv",
            'help' : "retrieve help on avilable comands"
        }

    @staticmethod
    def check_vol():
        print("Hello, world!")
        print(os.listdir('./'))
        # print(os.listdir('./shared/'))

        try:
            with open('./shared/test.txt', 'a') as f:
                f.write("time is {}\n".format(datetime.now().strftime("%d.%m.%Y, %H:%M:%S")))

            with open('./shared/test.txt', 'r') as f:
                for n in f.readlines():
                    print(n, end='')
        except:
            print("Could not read the file")


    def initialize(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.server.bind(("0.0.0.0", self.port))
                break
            except:
                self.port -= 1

        self.server.listen()
        print("Connected with {} port".format(self.port))

        while(True):
            self.client, addr = self.server.accept()
            print("Connection from", addr)
            self.client.send("Connection successful\n".encode())
            exit_flag = False
            while(True):
                self.client.send("Enter command\n".encode())
                msg = self.client.recv(1024).decode()[:-1].split(' ')
                if msg[0] not in self.commands:
                    self.client.send("Incorrect command\n".encode())
                else:
                    try:
                        exit_flag = self.commands[msg[0]](*msg[1:])
                    except Exception as e:
                        self.client.send("Error encountered while excecuting the command: {}\n".format(e).encode())
                
                if exit_flag:
                    break


    def exit(self):
        self.client.send("Connection is closing\n".encode())
        self.client.close()
        return True

    def args_check(self, one, two=5):
        print("Args entered: {} {}".format(one, two))
        return False

    def predict(self, file_name):
        
        path = PATH_NEW_DATA + file_name;
        if not os.path.isfile(path):
            self.client.send("File not found, please, ensure that you uploaded it to ./docker/share/inp_data folder\n".encode())
            return False
        
        df = pd.read_csv(path)
        df = snap2feat(df)
        df_t = self.predictor.pipe.transform(df)

        res = self.predictor.predict(df_t)
        
        self.untrained_data = self.untrained_data._append(df)
        np.savetxt(PATH_OUT_DATA + 'predictions_' + file_name, res.numpy())
        self.untrained_data.to_csv(PATH_OLD_DATA)
       
        self.client.send("Predictions has been saved to {}, please see the last 10 predictions below:\n{}\n".format(
            PATH_OUT_DATA + 'predictions_' + file_name, res[-100:]
            ).encode())
        self.client.send("Signal 0 means downward trend\nSignal 1 means upward trend\nSignal 2 means midward trend\n".encode())

        return False

    def train(self, epochs=5):
        if self.untrained_data.shape[0] > 0:
            res = self.predictor.train(self.untrained_data, int(epochs))
            self.client.send("{}\n".format(res).encode())
        else:
            self.client.send("No data to train on, skipping...\n".encode())
            return False
        
        self.untrained_data = pd.DataFrame([])
        if os.path.isfile(PATH_OLD_DATA):
            os.remove(PATH_OLD_DATA)
        
        return False

    def help(self):
        for n in self.commands.keys():
            self.client.send("{}\t\t{}\n".format(n, self.description[n]).encode())
        return False


