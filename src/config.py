import argparse
import logging
import os


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    
    parser.add_argument("-device_type", type=str, default='gpu')
    parser.add_argument("-gpu_id", type=int, default=1)
    parser.add_argument("-gpu_allocate_rate", type=float, default=0.98)
    parser.add_argument('-debug', type='bool', default=False, help='whether it is debug mode')
    parser.add_argument('-tune_embedding', type='bool', default=False, help="fine tune embedding")
    parser.add_argument('-test_only', type='bool', default=False, help='test_only: no need to run training process')
    parser.add_argument('-random_seed', type=int, default=1013, help='Random seed')
    parser.add_argument('-embedding_path', type=str, default="/home/tangmin/data/glove", help='Word embedding file')
    parser.add_argument("-emb_dim", type=int, default=300)
    parser.add_argument("-model_name", type=str, default='RSC')
    parser.add_argument("-use_pretrain_embedding", type='bool', default=True)    
    parser.add_argument("-log_file", type=str, default='')
    parser.add_argument("-num_epoch", type=int, default=100)
    parser.add_argument("-print_iter", type=int, default=1)
    parser.add_argument("-eval_iter", type=int, default=1000)
    parser.add_argument("-summary", type='bool', default=True)
    parser.add_argument("-fasttext", type='bool', default=True)
    parser.add_argument("-lower", type='bool', default=True)
    parser.add_argument("-reinforced", type='bool', default=False)
    parser.add_argument("-load_model", type='bool', default=False)
    parser.add_argument('-load_step', type=int, default=0)

    parser.add_argument("-dataname", type=str, default='IMDB')
    parser.add_argument("-n_classes", type=int, default=10)
    parser.add_argument("-rnn_type", type=str, default='bigru')
    parser.add_argument("-hidden_dim", type=int, default=200)
    parser.add_argument("-max_sent_len", type=int, default=80)
    parser.add_argument("-max_sent_num", type=int, default=30)
    parser.add_argument('-grad_clipping',type=float, default=10.0)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-optimizer", type=str, default='adam')
    parser.add_argument("-learning_rate", type=float, default=0.001)
    parser.add_argument("-input_keep_prob", type=float, default=0.8)
    parser.add_argument("-wd", type=float, default=0.0)

    parser.add_argument("-use_user_info", type='bool', default=False)
    parser.add_argument("-use_product_info", type='bool', default=False)
    parser.add_argument("-usr_coeff", type=float, default=0.1)
    parser.add_argument("-prd_coeff", type=float, default=0.1)

    return parser.parse_args()


def get_logger(config):
    logger = logging.getLogger('{}'.format(config.model_name))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.INFO)
    if config.log_file != '':
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger