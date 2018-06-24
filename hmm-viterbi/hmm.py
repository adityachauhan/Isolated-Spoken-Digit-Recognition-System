# =======================================================
# Hidden Markov Models with Gaussian emissions
# =======================================================

import numpy as np
from sklearn.mixture import GaussianMixture

class HMM:

    def __init__(self, num_states=3, num_mixtures=1, self_transp=0.9):

        self.num_states = num_states
        self.states = [GaussianMixture(n_components=num_mixtures, covariance_type='diag', 
            init_params='kmeans', max_iter=10) for state_id in range(self.num_states)]
        transp = np.diag((1-self_transp)*np.ones(self.num_states-1,),1) + np.diag(self_transp*np.ones(self.num_states,))
        self.log_transp = np.log(transp)
        

    def viterbi_decoding(self, obs):
        
        # Length of obs sequence
        T = obs.shape[0]
        
        # Precompute log output probabilities [num_states x T]
        log_outp = np.array([self.states[state_id].score_samples(obs).T for state_id in range(self.num_states)],  dtype = float)

        
        #varialbe to store and return the best state sequence
        path = np.zeros(T,int)
        #path = np.empty((T,), dtype='int')
        #print(np.shape(path))
        #initialse delta probability
        delta = np.zeros((self.num_states, T), dtype = float)
        phi = np.zeros((self.num_states, T), int)
        delta[: ,0] = initial_dist + log_outp[:, 0]
        phi[:, 0] = 0
        #for each time loop over all the states to add transition prob. to previous best probabiltiy
        #add emmision probability to the maximum result and also record the index of max previous
        #best probability+transition probability.
        for t in range(1, T):
            for s in range(self.num_states):
                delta[s, t] = np.max(delta[:, t-1] + self.log_transp[:, s])+log_outp[s, t]
                phi[s, t] = np.argmax(delta[:, t-1] + self.log_transp[:, s])
                


        #Backtrace to get the best state sequence
        
        path[T-1] = np.argmax(delta[:, T-1])
        for t in range(T-2, 0,-1):
            path[t] = phi[path[t+1],[t+1]]            

        max_prob = np.max(delta[:, T-1])
        return max_prob, path


