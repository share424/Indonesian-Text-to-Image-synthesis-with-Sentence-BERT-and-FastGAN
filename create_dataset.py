import pickle
from sentence_transformers import SentenceTransformer
import json
import argparse
import yaml

def encode(sbert, filename, output, dataset):
    with open(filename, 'rb') as f:
        filenames = pickle.load(f)

    captions = []
    for file in filenames:
        for data in dataset['dataset']:
            fn = data['filename']
            if file.endswith(fn[:-4]):
                caps = []
                for cap in data['captions']:
                    caps.append(cap['indo'])
                embdding = sbert.encode(caps)
                captions.append(embdding)

    with open(output, 'wb') as f:
        pickle.dump(captions, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('-cfg', '--config', help='configuration file in .yml', required=True)
    args = parser.parse_args()

    config = {}

    # load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file.read())
    
    config = config['DATASET']
    sbert = SentenceTransformer(config['SBERT'])
    with open(config['DATASET'], 'r') as f:
        dataset = json.loads(f.read())
    encode(sbert, config['TRAIN_FILENAMES'], config['TRAIN_OUTPUT'], dataset)
    encode(sbert, config['TEST_FILENAMES'], config['TEST_OUTPUT'], dataset)

