# This script performs spoken digit recognition using Hidden Markov Models


import numpy as np
import scipy.io.wavfile as wav
import os, pickle
import speechtech.frontend as feat


# =================================================================
# GLOBAL VARIABLES
# =================================================================
DIGITS = ['z', 'o', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_DIGITS = len(DIGITS)
DATA_DIR = 'data'
TRAIN_LIST = '{0}/flists/flist_train.txt'.format(DATA_DIR)
TEST_LIST = '{0}/flists/flist_test.txt'.format(DATA_DIR)

# HMM parameters
NUM_STATES = 10  # number of HMM states
NUM_MIXTURES = 3  # number of Gaussian mixtures per state
FEATURE_TYPE = 'mfcc'  # 'fbank' or 'mfcc'
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, "hmmfile_{0}_{1}states_{2}mix".format(FEATURE_TYPE, NUM_STATES, NUM_MIXTURES))


def eval_HMMs(hmm_set, file_list, feature_type='mfcc'):
    
    num_files = len(file_list)
    print('==== Evaluation feature type: {}'.format(feature_type))
    print('Number of utterances = {}'.format(num_files))
    target_labels = []
    rec_labels = []
    for n in range(num_files):
        fn = file_list[n]
        # Obtain the label for the data
        lab = os.path.splitext(os.path.basename(fn))[0][0]
        # Save target label in target_label
        target_labels = np.append(target_labels, lab)

        # Compute features
        fs_hz, signal = wav.read(fn)
        features = feat.compute_features[feature_type](signal, fs_hz)

        # -----------------------------------------------------------------
        # Use Viterbi decoding to get the log likelihood output by
        # each HMM. Take argmax to select the most likely HMM
        # -----------------------------------------------------------------
        #initialise array for log likelihood
        log_prob = []
        #populate array with log likelihood of each HMM
        for i in range(NUM_DIGITS):
            log_prob.append(hmm_set[i].viterbi_decoding(features)[0])

        # record label for maximum likelihood HMM
        rec = DIGITS[np.argmax(log_prob)]
        
        rec_labels = np.append(rec_labels, rec)

        print('{0}/{1} "{2}": lab = "{3}", rec = "{4}"'.format(n+1, num_files, fn, lab, rec))

    # Compute word error rate (for isolated digit recognition there are only substitution errors)
    WER = np.mean(target_labels != rec_labels)
    print('---- Word error rate using "{0}" is {1:0.2f}%\n'.format(feature_type, WER*100))

    return WER, target_labels, rec_labels



def main():

    # Load trained HMM set
    if not os.path.exists(MODEL_FILE):
        print('HMM file cannot be found: {0}'.format(MODEL_FILE))
        exit()
    with open(MODEL_FILE, 'rb') as f:
        hmm_set = pickle.load(f)
    print('Loaded HMM file: {0}'.format(MODEL_FILE))
    
    # Read test set list
    with open(TEST_LIST) as f:
        flist = f.readlines()
    flist = [x.strip() for x in flist]

    # Evaluation
    WER, target_labels, rec_labels = eval_HMMs(hmm_set, flist, feature_type=FEATURE_TYPE)



if __name__ == '__main__':
    main()

