from sentence_transformers import SentenceTransformer, losses, evaluation, models, InputExample
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pickle
import yaml
import numpy as np
import tqdm
import torch
import itertools
import json

train_score = [0]

def train_objective(score, epoch, steps):
    global train_score
    best_score = max(score, max(train_score))
    train_score.append(score)
    print('Epoch: {} - Step: {} - Score: {} - Best Score: {}'.format(epoch, steps, score, best_score))

def load_dataset(config):
    with open(config['DATASET'], 'r') as f:
        dataset = json.loads(f.read())

    positive_lower = config['POSTIVE']['START']
    positive_upper = config['POSTIVE']['END']

    negative_lower = config['NEGATIVE']['START']
    negative_upper = config['NEGATIVE']['END']

    train_examples = []

    captions = {}

    for data in dataset['dataset']:
        for cap in data['captions']:
            if data['class_id'] not in captions:
                captions[data['class_id']] = []
            captions[data['class_id']].append(cap['indo'])
    
    # create combinations of positive and negative examples
    for class_id in captions:
        # shuffle captions
        np.random.shuffle(captions[class_id])
        positive_pair_captions = itertools.combinations(captions[class_id], 2)
        positive_pair_captions = list(positive_pair_captions)[:700]
        for pair in positive_pair_captions:
            # create random value between positive_lower and positive_upper
            label = positive_lower + (np.random.rand() * np.abs(positive_upper - positive_lower))
            train_examples.append(InputExample(texts=[pair[0], pair[1]], label=np.float32(label)))
    
    # create pairs of negative examples from different classes from captions
    for class_id in captions:
        for i in range(600):
            target_caption_idx = np.random.randint(0, len(captions))
            while target_caption_idx == class_id:
                target_caption_idx = np.random.randint(0, len(captions))
            data_target = captions[target_caption_idx]
            caption_idx = np.random.randint(0, len(data_target))
            caption_target = data_target[caption_idx]
            cap_idx = np.random.randint(0, len(captions[class_id]))
            caption = captions[class_id][cap_idx]
            # create random value between negative_lower and negative_upper
            label = negative_lower + (np.random.rand() * np.abs(negative_upper - negative_lower))
            train_examples.append(InputExample(texts=[caption, caption_target], label=np.float32(label)))
    return train_examples

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-cfg', '--config', help='configuration file in .yml', required=True)
    
    args = parser.parse_args()

    config = {}

    # load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file.read())

    assert 'FINETUNE_SBERT' in config, 'Configuration file is not valid'

    np.random.seed(config['SEED'])

    config = config['FINETUNE_SBERT']

    # suffix = '.pkl'
    # if config['LOSS'] == 'triplet_loss':
    #     suffix = '_triplet.pkl'

    train_examples = load_dataset(config)

    np.random.shuffle(train_examples)

    # calculate validation amount
    n_val = int(np.floor(len(train_examples) * config['VALIDATION']))
    # split train and validation data
    validation_examples = train_examples[:n_val]
    train_examples = train_examples[n_val:]

    # create validation pair (ONLY work on cosine similarity dataset)
    # TODO: fix this to work on triplet dataset
    evaluator = None
    if config['LOSS'] == 'cosine_similarity':
        sentences1 = [tx.texts[0] for tx in validation_examples]
        sentences2 = [tx.texts[1] for tx in validation_examples]
        scores = [tx.label for tx in validation_examples]
        evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    
    # load the base model
    # model = SentenceTransformer(config['MODEL'])
    word_embedding_model = models.Transformer(config['MODEL'], max_seq_length=30)
    pooling_layer = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    nn1 = models.Dense(in_features=pooling_layer.get_sentence_embedding_dimension(), out_features=config['EMBEDDING_DIM'], activation_function=torch.nn.Tanh())
    # nn2 = models.Dense(in_features=512, out_features=256, activation_function=torch.nn.Tan())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_layer, nn1])
    # model.load_state_dict(torch.load('checkpoints\pretrain_sbert_v5_triplet\indo-sbert.pth'))
    # model.eval()
    print('n dataset:', len(train_examples))
    train_dataloader = DataLoader(train_examples, shuffle=config['SHUFFLE'], batch_size=config['BATCH_SIZE'])
    if config['LOSS'] == 'cosine_similarity':
        train_loss = losses.CosineSimilarityLoss(model)
    elif config['LOSS'] == 'triplet_loss':
        train_loss = losses.BatchHardTripletLoss(model=model, margin=config['TRIPLET_MARGIN'])
    else:
        raise ValueError('LOSS not implemented yet')

    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=config['EPOCHS'], 
        warmup_steps=config['WARMUP_STEP'], 
        show_progress_bar=True,
        output_path=config['OUTPUT_MODEL_PATH'],
        save_best_model=True,
        evaluator=evaluator,
        callback=train_objective
    )

    torch.save(model.state_dict(), config['FINAL_MODEL_PATH'])



