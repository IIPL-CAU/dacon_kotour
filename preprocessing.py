import os
import time
import h5py
import pickle
import logging
import numpy as np
from sklearn.model_selection import train_test_split
# Import custom modules
from utils import TqdmLoggingHandler, write_log

def preprocessing(args):

    start_time = time.time()

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #=============Data Load=============#
    #===================================#

    write_log(logger, 'Start preprocessing!')

    train = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    train_text, valid_text, train_label, valid_label = train_test_split(train['overview'].tolist(), 
                                                                        train['cat3'].tolist(), 
                                                                        test_size=0.2, random_state=42)

    test = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    test_text = test['overview'].tolist()