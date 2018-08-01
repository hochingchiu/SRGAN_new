import argparse
import os
from math import log10
import datetime

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform, TestPlay
from model import Generator


timenow = datetime.datetime.now().strftime("%Y%m%d-%H%M")
os.mkdir(timenow)
path1= timenow + '/benchmark_results'
os.mkdir(path1)
#path2= timenow+'/epochs'
#os.mkdir(path2)


UPSCALE_FACTOR = 2
MODEL_NAME = 'netG_epoch_2_100.pth'

results = {'Set5': {'psnr': [], 'ssim': []}, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR).eval()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

test_set = TestPlay('data/play', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = timenow + '/benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

for image_name, lr_image in test_bar:
    image_name = image_name[0]
    lr_image = Variable(lr_image, volatile=True)
   # hr_image = Variable(hr_image, volatile=True)
    if torch.cuda.is_available():
        lr_image = lr_image.cuda()
   #     hr_image = hr_image.cuda()

    sr_image = model(lr_image)

    image = sr_image
    utils.save_image(image, 'output.png')

    # save psnr\ssim
    results[image_name.split('_')[0]]['psnr'].append(psnr)
    results[image_name.split('_')[0]]['ssim'].append(ssim)

