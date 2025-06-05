import numpy as np
import sentence_transformers
import numpy.random as npr
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm
import os
tqdm.pandas()

class DPComet():
    '''
    DPComet class: this class implements the DPComet mechanism, which is a
    differential privacy mechanism for embedding spaces. It uses the
    Contriever model from the Sentence Transformers library to encode text
    into embeddings, and it implements two noise pulling methods: CMP and
    Mahalanobis. The class also provides methods to create, load, and dump
    the embeddings matrix, as well as to encode text.
    '''
    def __init__(self, spaceQueries:pd.Series, model: str = 'bert-base-nli-mean-tokens') -> None:
        '''
        Constructor for the DPComet class.
        :param spaceQueries: pd.Series, the collection of queries to be encoded
        :param model: str, the name of the Sentence Transformer model to be used
        '''
        self.name:str = f'Contriever'
        self.model:object = sentence_transformers.SentenceTransformer(model)
        # print encoding dimensions
        print(f'Encoding dimensions: {self.model.get_sentence_embedding_dimension()}')
        self.set_self_embMatrix(spaceQueries)
        self.lam:float = 0.75
        self.epsilon:float = None
        cov_mat:np.array = np.cov(self.embMatrix.T, ddof=0)
        sigma:np.array = cov_mat/ np.mean(np.var(self.embMatrix.T, axis=1))
        self._sigmaLoc:float = sqrtm(self.lam * sigma + (1 - self.lam) * np.eye(self.embMatrix.shape[1]))
        self._rescaleFactor:float = np.sqrt(self.embMatrix.shape[1])

    def set_self_embMatrix(self, collection:pd.Series) -> None:
        '''
        set_self_embMatrix method: this method is used to set the embeddings matrix
        :param collection: pd.Series, the collection of queries to be encoded
        This method checks if the embeddings matrix already exists in the specified path.
        If it exists, it loads the embeddings matrix from the file. If it does not exist,
        it creates the embeddings matrix by encoding the collection of queries and saves it to a file.
        '''
        #if it does not exist the embMatrix.npy file, create it

        if not os.path.exists(f'./config/test/embeddingSpace/embMatrix_{self.name}_NLP.npy'): #_IR
            self.create_embMatrix(collection)
        else:
            self.embMatrix = np.load(f'./config/test/embeddingSpace/embMatrix_{self.name}_NLP.npy') #_IR

    def create_embMatrix(self, collection:pd.Series) -> None:
        '''
        create_embMatrix method: this method is used to create the embeddings matrix
        :param collection: pd.Series, the collection of queries to be encoded
        This method encodes the collection of queries using the Sentence Transformer model
        and saves the embeddings matrix to a file.
        '''
        self.embMatrix = collection.progress_apply(lambda x: self.model.encode(x))
        self.embMatrix = np.array(self.embMatrix.tolist())
        self.dump_embMatrix()

    def get_embMatrix(self) -> np.array:
        '''
        get_embMatrix method: this method is used to get the embeddings matrix
        :return: np.array, the embeddings matrix
        This method returns the embeddings matrix.
        '''
        return self.embMatrix

    def set_epsilon(self, epsilon) -> None:
        '''
        set_epsilon method: this method is used to set the epsilon value
        :param epsilon: float, the epsilon value to be set
        This method sets the epsilon value for the DPComet mechanism.
        '''
        self.epsilon = epsilon

    def dump_embMatrix(self, compressed: bool = False) -> None:
        if compressed:
            np.savez_compressed('./config/embMatrix.npz', self.embMatrix)
        else:
        #save the embeddings matrix
            np.save(f'./config/embMatrix_{self.name}.npy', self.embMatrix)
        
    def encode(self, text) -> np.array:
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