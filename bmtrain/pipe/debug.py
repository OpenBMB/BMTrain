from bmtrain.global_var import config
import logging

def get_logger(rank):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.DEBUG)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    fh = logging.FileHandler(f'pipe_{rank}.log',mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
