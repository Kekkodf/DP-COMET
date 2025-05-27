from src.utils import mylogger as MyLogger
import ir_datasets
from src.dp_comet import DPComet
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

tqdm.pandas()

logger = MyLogger.create()
#space = ir_datasets.load('msmarco-passage/train')
space = pd.read_csv('./config/test/textSpace/tweet_pool.csv')
#rename columns to match the expected format
space = space.rename(columns={'Tweet description': 'query'})
#drop Name column
space = space.drop(columns=['Name'])
#add qid column
space['qid'] = space.index
#order columns to match the expected format
space = space[['qid', 'query']]
space_queries = space

queries = pd.read_csv('./tweets-df.csv')
#rename columns to match the expected format
queries = queries.rename(columns={'id': 'qid', 'text': 'query'})
logger.info(f'Queries in the dataset: {queries.shape[0]}')
contextual = DPComet(spaceQueries=space_queries['query'], model='facebook/contriever-msmarco')
logger.info('Mechanism object created')


if __name__ == '__main__':
    
    space_embeddings_matrix = contextual.get_embMatrix()
    logger.info(f'Queries in the space: {space_embeddings_matrix.shape[0]}')
    logger.info(f'Embedding dimensions: {space_embeddings_matrix.shape[1]}')
    #space_embeddings_matrix = space_queries['query'].progress_apply(lambda x: contextual.encode(x))
    #space_embeddings_matrix = np.array(space_embeddings_matrix.tolist())
    #logger.info(f'Queries in the space: {space_embeddings_matrix.shape[0]}')
    #logger.info(f'Embedding dimensions: {space_embeddings_matrix.shape[1]}')

    queries_embeddings_matrix = queries['query'].progress_apply(lambda x: contextual.encode(x))
    queries_embeddings_matrix = np.array(queries_embeddings_matrix.tolist())
    logger.info(f'Queries in the queries: {queries_embeddings_matrix.shape[0]}')
    logger.info(f'Embedding dimensions: {queries_embeddings_matrix.shape[1]}')

    #add noise to the embeddings of the space
    for epsilon in [1, 5, 10, 12.5, 15, 17.5, 20, 25, 30, 50]:
        df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
        contextual.set_epsilon(epsilon)
        logger.info(f'Noise level: {epsilon}')
        for i in range(1):
            logger.info(f'Iteration: {i}')
            query_embs = queries_embeddings_matrix.copy()
            query_embs = queries_embeddings_matrix + contextual.pullNoiseCMP()
            #logger.info(f'Noise added to the queries embeddings: {query_embs.shape}')

            #compute the cosine similarity between the space and the queries
            similarity = np.dot(space_embeddings_matrix, query_embs.T)
            #logger.info(f'Similarity matrix: {similarity.shape}')

            #get the top most similar queries
        
            top_k_indices = np.argmax(similarity, axis=0)
            #get the top k queries
            top_k_queries = space_queries.iloc[top_k_indices]
            temp = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'sentiment', 'epsilon', 'mechanism'])
            temp['id'] = queries['qid']
            temp['text'] = queries['query']
            temp['obfuscatedText'] = top_k_queries['query'].values
            temp['epsilon'] = epsilon
            temp['mechanism'] = 'ContextualCMP'
            temp['sentiment'] = queries['sentiment'].values

            df = pd.concat([df, temp])

        df.sort_values(by='id', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        os.makedirs('results/NLP/ContextualCMP', exist_ok=True)
        df.to_csv('results/NLP/ContextualCMP/obfuscatedText_ContextualCMP_{}.csv'.format(epsilon), index=False)
    logger.info('Finished ContextualCMP')

    for epsilon in [1, 5, 10, 12.5, 15, 17.5, 20, 25, 30, 50]:
        df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'sentiment', 'epsilon', 'mechanism'])
        contextual.set_epsilon(epsilon)
        logger.info(f'Noise level: {epsilon}')
        for i in range(1):
            logger.info(f'Iteration: {i}')
            query_embs = queries_embeddings_matrix.copy()
            query_embs = queries_embeddings_matrix + contextual.pullNoiseMhl()
            #logger.info(f'Noise added to the queries embeddings: {query_embs.shape}')

            #compute the cosine similarity between the space and the queries
            similarity = np.dot(space_embeddings_matrix, query_embs.T)
            #logger.info(f'Similarity matrix: {similarity.shape}')

            #get the top most similar queries
        
            top_k_indices = np.argmax(similarity, axis=0)
            #get the top k queries
            top_k_queries = space_queries.iloc[top_k_indices]
            temp = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
            temp['id'] = queries['qid']
            temp['text'] = queries['query']
            temp['obfuscatedText'] = top_k_queries['query'].values
            temp['epsilon'] = epsilon
            temp['mechanism'] = 'ContextualMhl'
            temp['sentiment'] = queries['sentiment'].values

            df = pd.concat([df, temp])

        df.sort_values(by='id', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        os.makedirs('results/NLP/ContextualMhl', exist_ok=True)
        df.to_csv('results/NLP/ContextualMhl/obfuscatedText_ContextualMhl_{}.csv'.format(epsilon), index=False)
    logger.info('Finished ContextualMhl')