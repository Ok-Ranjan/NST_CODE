import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
import torch
from pathlib import Path
from utils.utils import *
from torch.utils.data import DataLoader

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default=r'C:\Users\ranja\Desktop\NST_CODE\content_data_demo', help='Location of CONTENT dataset')
    parser.add_argument('--style_dir', type=str, default=r'C:\Users\ranja\Desktop\NST_CODE\style_data_demo', help='Location of STYLE dataset')
    parser.add_argument('--vgg', type=str, default='', help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')

    parser.add_argument('--final_size', type=int, default=256, help="Size of final image")
    parser.add_argument('--content_size', type=int, default=512, help="Size of content image")
    parser.add_argument('--style_size', type=int, default=512, help="Size of style image")
    parser.add_argument('--crop', action='store_true', default=True, help="Crop Image")
   
    parser.add_argument('--batch_size', type=int, default=4, help="Set the batch size")
    return parser.parse_args()

def main():
    args = parse_argument()

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}: {value}\n")

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, transform=content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, transform=style_transform)

    content_dataloader = DataLoader(content_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)
    style_dataloader = DataLoader(style_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=True)
    
    print("Number of batches in content dataset", len(content_dataloader))
    print("Number of batches in style dataset", len(style_dataloader))

    for batch in style_dataloader:
        print(batch.shape)



if __name__ == '__main__':
    main()