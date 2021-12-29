import torch
from models import Generator, INCEPTION_V3
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from torch.autograd import Variable
from torchvision import utils as vutils
import yaml

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-cfg', '--config', help='configuration file in .yml', required=True)
    parser.add_argument('--input', type=str, help='string input', required=True)

    args = parser.parse_args()

    config = {}

    # load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file.read())

    assert 'TEST' in config, 'Configuration file is not valid'
    config = config['TEST']

    device = torch.device('cuda:0')

    print('Loading generator...')
    netG = Generator(
        ngf=config['GENERATOR_SIZE'], 
        nz=config['Z_SIZE'], 
        nc=3, 
        im_size=config['IMAGE_SIZE'], 
        t_dim=config['TEXT_EMB'], 
        emb_dim=config['CA_DIM']
    )
    checkpoint = torch.load(config['GAN'], map_location=lambda a,b: a)
    checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
    netG.load_state_dict(checkpoint['g'])
    netG.to(device)
    netG.eval()

    print('Loading sbert...')
    sbert = SentenceTransformer(config['SBERT'])
    sbert.to(device)
    text_embeddings = sbert.encode(args.input)
    text_embeddings = Variable(torch.FloatTensor(text_embeddings))
    text_embeddings = text_embeddings.unsqueeze(0)
    text_embeddings.to(device)
    text_embeddings = text_embeddings.repeat(4, 1)

    noise = torch.Tensor(4, 100).normal_(0, 1).to(device)
    generated_images, _, _ = netG(noise, text_embeddings.to(device))

    vutils.save_image(generated_images[0].add(1).mul(0.5), config['OUTPUT'], nrow=4)