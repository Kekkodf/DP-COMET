import os
import sys
import ir_datasets
import pandas as pd

from src.utils import mylogger

if __name__ == '__main__':
    logger = mylogger.create('main_logger')
    df = pd.read_csv('data/tweet.csv')
    df.columns = ['author', 'text']
    #print number of unique authors and number of tweets
    logger.info(f'Number of unique authors: {df["author"].nunique()}')
    logger.info(f'Number of tweets: {df.shape[0]}')
    #print average length of tweets
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    logger.info(f'Average length of tweets: {df["text_length"].mean()}')
    
    