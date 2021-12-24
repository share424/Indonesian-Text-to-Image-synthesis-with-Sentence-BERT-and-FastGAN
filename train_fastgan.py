import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import argparse
import random
from tqdm import tqdm

from models import INCEPTION_V3, weights_init, UnconditionalDiscriminator, ConditionalDiscriminator, Generator
from operation import TextDataset, copy_G_params, load_params, get_dir, compute_inception_score
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
import numpy as np
import yaml



#torch.backends.cudnn.benchmark = True


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, percept, label="real", c_code=None):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, c_code=c_code, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part, err
    else:
        pred = net(data, label, c_code=c_code)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item(), err
        

def train(config):
    total_iterations = config['ITERATIONS']
    checkpoint = config['CHECKPOINT']
    batch_size = config['BATCH_SIZE']
    im_size = config['IMAGE_SIZE']
    ndf = config['DISCRIMINATOR_SIZE']
    ngf = config['GENERATOR_SIZE']
    nz = config['Z_SIZE'] # latent vector size
    nlr = config['LR']
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = True
    dataloader_workers = 8
    current_iteration = 0
    save_interval = config['SAVE_INTERVAL']
    saved_model_folder, saved_image_folder = get_dir(config)
    emb_dim = config['CA_DIM']
    t_dim = config['TEXT_EMB']
    conditional = config['CONDITIONAL']
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)

    dataset_path = config['DATASET']
    dataset = TextDataset(
        dataset_path,
        split='train',
        embedding_type='sbert',
        transform=trans,
        train=True
    )

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
    
    #from model_s import Generator, Discriminator
    print('Initialize Models')
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size, t_dim=t_dim, emb_dim=emb_dim)
    netG.apply(weights_init)

    if conditional:
        netD = ConditionalDiscriminator(ndf=ndf, im_size=im_size, emb_dim=emb_dim)
    else:
        netD = UnconditionalDiscriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    inception = INCEPTION_V3()

    netG.to(device)
    netD.to(device)
    inception.to(device)
    inception.eval()

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1).to(device)
    fixed_image, _, fixed_txt_embedding, _ = next(dataloader)
    vutils.save_image(fixed_image.add(1).mul(0.5), saved_image_folder+'/ref.jpg', nrow=5)
    
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != '':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    
    print('Start Training')
    predictions = []
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image, wrong_image, txt_embedding, key = next(dataloader)
        real_image = real_image.to(device)
        wrong_image = wrong_image.to(device)
        txt_embedding = txt_embedding.to(device)
        
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images, mu, logvar = netG(noise, txt_embedding)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        
        ## 2. train Discriminator
        netD.zero_grad()

        c_code = None
        if conditional:
            c_code = mu.detach()

        err_dr, rec_img_all, rec_img_small, rec_img_part, err_real = train_d(netD, real_image, percept, label="real", c_code=c_code)
        _, err_fake = train_d(netD, [fi.detach() for fi in fake_images], percept, label="fake", c_code=c_code)
        if conditional:
            _, err_wrong = train_d(netD, wrong_image, percept, label="wrong", c_code=c_code)
        
        # weighted sum
        # err = err_fake + err_wrong + (err_real * 0.5)
        # err.backward()
        
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake", c_code=c_code)
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        pred = inception(fake_images[0].detach())
        predictions.append(pred.data.cpu().numpy())

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            predictions = np.concatenate(predictions, 0)
            mean, std = compute_inception_score(predictions, 10)
            print("\nGAN: loss d: %.5f    loss g: %.5f    inception mean: %.5f"%(err_dr, -err_g.item(), mean))
            predictions = []

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                generated_images, _, _ = netG(fixed_noise, fixed_txt_embedding)
                vutils.save_image(generated_images[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=5)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration, nrow=5)
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument('-cfg', '--config', help='configuration file in .yml', required=True)

    args = parser.parse_args()

    config = {}

    # load configuration file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file.read())

    assert 'TRAIN_FASTGAN' in config, 'Configuration file is not valid'

    train(config['TRAIN_FASTGAN'])
