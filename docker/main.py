from datetime import datetime
from server import Server
from dotenv import dotenv_values
import os

# params = {
#         'PATH_MODEL' : './shared/model',
#         'PATH_OLD_DATA' : './shared/past_data.csv',
#         'PATH_NEW_DATA' : './shared/inp_data/',
#         'PATH_TEST_DATA' : './shared/test_tensors.t',
#         'PATH_OUT_DATA' : './shared/out_data/',
#         'MAX_TEST_SAMPLES' : 1e4,
#     }
    
def create_folders():
    files = [PATH_NEW_DATA, PATH_OUT_DATA]
    for path in files:
        if not os.path.isdir(path):
            os.mkdir(path, mode = 777)
    

if __name__ == '__main__':
    params = dotenv_values('./config.env')
    params['MAX_TEST_SAMPLES'] = int(params['MAX_TEST_SAMPLES'])
    globals().update(params)
    create_folders()
    port = 9990
    serv = Server(port, params)
    serv.check_vol()
    serv.initialize()


