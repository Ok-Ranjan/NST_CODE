import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import *
from utils.models import *
from tqdm import tqdm  # comman library use to tracking loop

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default=r'C:\Users\ranja\Desktop\NST_CODE\content_data_demo', help='Location of CONTENT dataset')
    parser.add_argument('--style_dir', type=str, default=r'C:\Users\ranja\Desktop\NST_CODE\style_data_demo', help='Location of STYLE dataset')
    parser.add_argument('--vgg', type=str, default='vgg_normalised.pth', help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')

    parser.add_argument('--final_size', type=int, default=256, help="Size of final image")
    parser.add_argument('--content_size', type=int, default=512, help="Size of content image")
    parser.add_argument('--style_size', type=int, default=512, help="Size of style image")
    parser.add_argument('--crop', action='store_true', default=True, help="Crop Image")
   
    parser.add_argument('--batch_size', type=int, default=4, help="Set the batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content weight')
    parser.add_argument('--style_weight', type=float, default=10, help='Style weight')
    parser.add_argument('--log_interval', type=int, default=5, help="Log Interval")
    return parser.parse_args()

def main():
    args = parse_argument()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    
    # scheduler make sure that learning rate is not fixed, change accroding to epoch for better performation.
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    print("Training...")
    
    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))
        
        running_loss = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # 1. every image first go through Encoder
            c_features = encoder(content_batch) # 4 features
            s_features = encoder(style_batch) # 4 features

            # print(len(c_features))
            # print(len(s_features))
            # print(c_features[0].shape)

            # 2. out of encoder => AdaIn layer
            t = adaptive_instance_normalization(c_features[-1], s_features[-1])

            # out of Adain Layer => decoder
            g = decoder(t) # g is supposed to be output

            g_features = encoder(g)

            loss_c = mse_loss(g_features[-1], t) * args.content_weight

            loss_s = 0
            for g_f, s_f in zip(g_features, s_features):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            
            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward() # calculate gradients based on new loss
            optimizer.step() # update paarmeters using gradients

            progress_bar.set_description(f'Iter {epoch+1}: Loss: {running_loss:4f}, Content Loss: {running_closs:4f}, Style loss: {running_sloss:4f}')

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()
        
        scheduler.step() # change the lr

        running_loss /= len(content_dataloader)
        running_closs /= len(content_dataloader)
        running_sloss /= len(content_dataloader)
        
        if (epoch+1) % args.log_interval == 0:
            tqdm.write(f'Iter {epoch+1}: Loss: {running_loss:4f}, Content Loss: {running_closs:4f}, Style loss: {running_sloss:4f}')














if __name__ == '__main__':
    main()