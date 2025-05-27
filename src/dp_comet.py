import numpy as np
import sentence_transformers
import numpy.random as npr
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm
import os
tqdm.pandas()

class DPComet():
    def __init__(self, spaceQueries:pd.Series, model: str = 'bert-base-nli-mean-tokens'):
        self.name = f'Contriever'
        self.model = sentence_transformers.SentenceTransformer(model)
        # print encoding dimensions
        print(f'Encoding dimensions: {self.model.get_sentence_embedding_dimension()}')
        self.set_self_embMatrix(spaceQueries)
        self.lam = 0.75
        self.epsilon = None
        cov_mat = np.cov(self.embMatrix.T, ddof=0)
        sigma = cov_mat/ np.mean(np.var(self.embMatrix.T, axis=1))
        self._sigmaLoc = sqrtm(self.lam * sigma + (1 - self.lam) * np.eye(self.embMatrix.shape[1]))
        self._rescaleFactor = np.sqrt(self.embMatrix.shape[1])

    def set_self_embMatrix(self, collection:pd.Series):
        #if it does not exist the embMatrix.npy file, create it

        if not os.path.exists(f'./config/test/embeddingSpace/embMatrix_{self.name}_NLP.npy'): #_IR
            self.create_embMatrix(collection)
        else:
            self.embMatrix = np.load(f'./config/test/embeddingSpace/embMatrix_{self.name}_NLP.npy') #_IR

    def create_embMatrix(self, collection:pd.Series):
        self.embMatrix = collection.progress_apply(lambda x: self.model.encode(x))
        self.embMatrix = np.array(self.embMatrix.tolist())
        self.dump_embMatrix()

    def get_embMatrix(self):
        return self.embMatrix

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def dump_embMatrix(self, compressed: bool = False):
        if compressed:
            np.savez_compressed('./config/embMatrix.npz', self.embMatrix)
        else:
        #save the embeddings matrix
            np.save(f'./config/embMatrix_{self.name}.npy', self.embMatrix)
        
    def encode(self, text):
        return self.model.encode(text)
    
    def pullNoiseCMP(self) -> np.array:
        '''
        method pullNoise: this method is used to pull noise accordingly 
        to the definition of the CMP mechanism, see BibTeX ref

        : return: np.array the noise pulled

        Usage example:
        (Considering that the Mechanism Object mech1 has been created 
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''
        N: np.array = self.epsilon * npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]),
            np.eye(self.embMatrix.shape[1])) #pull noise from a multivariate normal distribution
        X: np.array = N / np.sqrt(np.sum(N ** 2)) #normalize the noise
        Y: np.array = npr.gamma(
            self.embMatrix.shape[1],
            1 / (self.epsilon*self._rescaleFactor)) #pull gamma noise
        Z: np.array = Y * X #compute the final noise
        return Z
    
    def pullNoiseMhl(self) -> np.array:
        '''
        method pullNoise: this method is used to pull noise accordingly 
        to the definition of the Mahalanobis mechanism, see BibTeX ref
        : return: np.array the noise pulled
        Usage example:
        (Considering that the Mechanism Object mech1 has been created 
        as in the example of the __init__ method)
        >>> mech1.pullNoise()
        '''

        N: np.array = npr.multivariate_normal(
            np.zeros(self.embMatrix.shape[1]), 
            np.eye(self.embMatrix.shape[1])
            )
        X: np.array = N / np.sqrt(np.sum(N ** 2))
        X: np.array = np.dot(self._sigmaLoc, X)
        X: np.array = X / np.sqrt(np.sum(X ** 2))
        Y: np.array = npr.gamma(
            self.embMatrix.shape[1], 
            1 / (self.epsilon * self._rescaleFactor)
            )
        Z: np.array = Y * X
        return Z