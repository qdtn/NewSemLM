import argparse
from liir1.nlp.semlm.exp.ConfigReader import read_config
from liir1.nlp.semlm.models.oneone.Model import trainSemLM11
from liir1.nlp.semlm.models.twoone.Model import trainSemLM21
from liir1.nlp.semlm.models.twotwo.Model import trainSemLM22

__author__ = 'quynhdo'




def trainModel11(config_path):
    cfg = read_config(config_path)

    trainSemLM11(cfg['train'], cfg['valid'], cfg['we_dict'], cfg['dep'],
                 cfg['hidden_size'], cfg['batch_size'], cfg['save_folder'], load_dt=cfg['load_data'])

def trainModel21(config_path):
    cfg = read_config(config_path)

    trainSemLM21(cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],load_dt=cfg['load_data'])


def trainModel22(config_path):
    cfg = read_config(config_path)

    trainSemLM22(cfg['train'], cfg['valid'], cfg['we_dict1'], cfg['we_dict2'], cfg['dep'],
                 cfg['hidden_size1'], cfg['hidden_size2'],cfg['batch_size'], cfg['save_folder'],load_dt=cfg['load_data'])

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("exp_config", help="exp configuration file")

    parser.add_argument('--11', dest='mdl11', action='store_true')
    parser.add_argument('--21', dest='mdl21', action='store_true')
    parser.add_argument('--22', dest='mdl22', action='store_true')

    args = parser.parse_args()


    if args.mdl11:
        trainModel11(args.exp_config)
    if args.mdl21:
        trainModel21(args.exp_config)

    if args.mdl22:
        trainModel22(args.exp_config)




    #trainModel22("/Users/quynhdo/Documents/PhDFinal/workspace/NewSemLM/config/exp.config")