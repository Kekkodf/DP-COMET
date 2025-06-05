from src.utils import mylogger as MyLogger
import ir_datasets
from src.dp_comet import DPComet
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import argparse
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

mapper = {'medline04': 'medline/2004/trec-genomics-2004',
          'dl19': 'msmarco-passage/trec-dl-2019/judged',
          'dl20': 'msmarco-passage/trec-dl-2020/judged'}

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description='DP-COMET Obfuscation Script')
parser.add_argument('--epsilons', type=float, nargs='+', default=[1, 5, 10, 12.5, 15, 17.5, 20, 50],
                    help='List of epsilon values for differential privacy')
parser.add_argument('--iterations', type=int, default=50, help='Number of iterations for obfuscation')
parser.add_argument('--mechanism', type=str, choices=['CMP', 'Mhl'], default='CMP', help='Mechanism to use for obfuscation')
parser.add_argument('--collection', type=str, choices=['medline04', 'dl19', 'dl20'], default='medline04',
                    help='Collection to use for the obfuscation process')
# Parse the arguments
args = parser.parse_args()

logger = MyLogger.create()
space = ir_datasets.load('msmarco-passage/train')  # Load the collection based on the argument
space_queries = space.queries_iter()
space_queries = pd.DataFrame(space_queries, columns=['qid', 'query'])

queries = ir_datasets.load(mapper[args.collection]) #msmarco-passage/trec-dl-2020/judged, msmarco-passage/trec-dl-2019/judged, medline/2004/trec-genomics-2004
queries = queries.queries_iter()
#remove columns need and context
if args.collection == 'medline04':
    queries = [(q.query_id, q.title) for q in queries]
queries = pd.DataFrame(queries, columns=['qid', 'query'])
logger.info(f'Queries in the dataset: {queries.shape[0]}')
contextual = DPComet(spaceQueries=space_queries['query'], model='facebook/contriever-msmarco')
logger.info('Mechanism object created')

if __name__ == '__main__':
    
    space_embeddings_matrix = contextual.get_embMatrix()
    logger.info(f'Queries in the space: {space_embeddings_matrix.shape[0]}')
    logger.info(f'Embedding dimensions: {space_embeddings_matrix.shape[1]}')

    queries_embeddings_matrix = queries['query'].progress_apply(lambda x: contextual.encode(x))
    queries_embeddings_matrix = np.array(queries_embeddings_matrix.tolist())
    logger.info(f'Queries in the query space: {queries_embeddings_matrix.shape[0]}')
    logger.info(f'Embedding dimensions: {queries_embeddings_matrix.shape[1]}')
    if args.mechanism == 'CMP':
        #add noise to the embeddings of the space
        print('Starting the obfuscation process with DP-COMET-CMP')
        t_0 = time.time()
        for epsilon in args.epsilons:
            df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
            contextual.set_epsilon(epsilon)
            logger.info(f'Noise level: {epsilon}')
            for i in trange(args.iterations, desc=f'Obfuscating queries with epsilon {epsilon:.1f}'):
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
                temp = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
                temp['id'] = queries['qid']
                temp['text'] = queries['query']
                temp['obfuscatedText'] = top_k_queries['query'].values
                temp['epsilon'] = epsilon
                temp['mechanism'] = 'DP-COMET-CMP'

                df = pd.concat([df, temp])

            df.sort_values(by='id', inplace=True, ascending=True)
            df.reset_index(drop=True, inplace=True)
            os.makedirs(f'results/{args.collection}/DP-COMET-CMP', exist_ok=True)
            df.to_csv(f'results/{args.collection}/DP-COMET-CMP/obfuscatedText_DP-COMET-CMP_{epsilon}.csv', index=False)
        logger.info('Finished DP-COMET-CMP')
        #format time in seconds, 2 decimal places
        print(f'Finished DP-COMET-CMP in time: {time.time() - t_0:.2f} seconds')
    elif args.mechanism == 'Mhl':
        print('Starting the obfuscation process with DP-COMET-Mhl')
        t_0 = time.time()
        for epsilon in args.epsilons:
            df = pd.DataFrame(columns=['id', 'text', 'obfuscatedText', 'epsilon', 'mechanism'])
            contextual.set_epsilon(epsilon)
            logger.info(f'Noise level: {epsilon}')
            for i in trange(args.iterations, desc=f'Obfuscating queries with epsilon {epsilon:.1f}'):
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
                temp['mechanism'] = 'DP-COMET-Mhl'

                df = pd.concat([df, temp])

            df.sort_values(by='id', inplace=True, ascending=True)
            df.reset_index(drop=True, inplace=True)
            os.makedirs(f'results/{args.collection}/DP-COMET-Mhl', exist_ok=True)
            df.to_csv(f'results/{args.collection}/DP-COMET-Mhl/obfuscatedText_DP-COMET-Mhl_{epsilon}.csv', index=False)
        logger.info('Finished DP-COMET-Mhl')

        print(f'Finished DP-COMET-Mhl in time: {time.time() - t_0:.2f} seconds')
    else:
        logger.error('Invalid mechanism selected. Choose either "CMP" or "Mhl".')
        raise ValueError('Invalid mechanism selected. Choose either "CMP" or "Mhl".')