# Isolated-Spoken-Digit-Recognition-System
The basic Hidden Markov Model based recognition system with GMM emissions.

gmm_demo.py is a model based on Gaussian Markov to test the error rate of two frameworks (MFCC and Fbank) for isolated digit recognition.

hmm.py is a Viterbi decoder that returns the best state sequence and maximum likelihood of the sequence.

hmm_train.py is used to train the model and get transition probabilities and obervation sequence.

hmm_eval.py is used to test the developed model and get the error rate for two different frameworks(MFCC and Fbank) for isolated digit recognition.

Data folder contains two zip files that contains the testing and training datasets.

The number of states in this model can vary from 1 to 10 thatcan be changed in code itself for training and testing.

To check for correctness of viterbi decoder in hmm.py the test_viterbi.py is available.
