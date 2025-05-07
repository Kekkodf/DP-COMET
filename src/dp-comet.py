import numpy as np
import transformers
import numpy.random as npr
from scipy.linalg import sqrtm
import pandas as pd
from tqdm import tqdm
import os
tqdm.pandas()

class Comet:
    def __init__(self, model_name:str='bert-base-uncased', spaceQueries:pd.DataFrame=None):
        
        self.model_name = model_name
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.set_self_embMatrix(spaceQueries)
    
    def set_self_embMatrix(self, spaceQueries:pd.DataFrame):
        #if it does not exist the embMatrix.npy file, create it

        if not os.path.exists(f'./config/embMatrix_{self.name}.npy'):
            self.create_embMatrix(collection)
        else:
            self.embMatrix = np.load(f'./config/embMatrix_{self.name}.npy')
    