import argparse
import torch
import models
import os
from data import testsets

parser = argparse.ArgumentParser(description='Frame Interpolation Evaluation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--net', type=str, default='EDC')
parser.add_argument('--dataset', type=str, default='Ucf101_quintuplet')
parser.add_argument('--metrics', nargs='+', type=str, default=['PSNR', 'SSIM'])
parser.add_argument('--checkpoint', type=str, default='ckpt.pth')
parser.add_argument('--data_dir', type=str, default='D:\\')
parser.add_argument('--out_dir', type=str, default='eval_results')

# model parameters
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    model = getattr(models, args.net)(args).cuda()

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    print('Testing on dataset: ', args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)
    if args.dataset.split('_')[0] in ['VFITex', 'Ucf101', 'Davis90']:
        db_folder = args.dataset.split('_')[0].lower()
    else:
        db_folder = args.dataset.lower()
    test_db = getattr(testsets, args.dataset)(os.path.join(args.data_dir, db_folder))
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    test_db.eval(model, metrics=args.metrics, output_dir=test_dir)



if __name__ == "__main__":
    main()
