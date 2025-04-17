#create a wrapper for the logging module
import logging
import os

def create(name:str = 'mylogger') -> logging.Logger:
    try:   
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        log_dir = './logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f'{name}.log')
        open(log_file, 'w').close()
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        logger.info('Logger successfully created')
        return logger
    except Exception as e:
        print(f'Error creating logger: {e}')
        return None