import torch.nn as nn
import torch

# define the standard VGG-19 Encoder architecture (Convolutional backbone of VGG-19)
class VGGEncoder(nn.Module):
    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        # Define the complete VGG-19 structure so the weights file fits perfectly
            # all Relu are the checkpoints inside the brain of the VGG network. Each checkpoint sees our image at a different level of details.
            # VGG devide into 5 major blocks and each block have multiple convolutional layers.
            # as we go from Block 1 to 5: the network zoom out, shrinking the image size but looking at bigger patterns. 
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)), # 1x1 convolution acts as a learnable color space or normalization transformation, before feature extraction starts.
            nn.ReflectionPad2d((1, 1, 1, 1)), #  Reflection padding mirrors the pixel values at the borders, instead of black border
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(), # relu1-1 (index: 3)

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(), # relu1-2


            nn.MaxPool2d((2, 2),  (2, 2), (0, 0), ceil_mode=True), # ceil_mode -> handles odd-dimension spatial feature, ensure feature edge details are not cut off. 
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(), # relu2-1 (index: 10)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(), # relu2-2

            # relu1 & relue2 => capture the fine texture of style painting | they see the tiny details.

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(), # relu3-1 (index: 17)
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), # relu3-4
            
            # relu3 => it sees complex textures and geometric arrangments. and start recognizing wave patterns, repetitive tile shapes, and shading. 

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(), # relu4-1 (index: 30), this is the last layer used - Content layer

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu4-4

            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(), # relu5-4
        )
        
        # load the pre-trained weights
        self.vgg.load_state_dict(torch.load(vgg_path))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31]) # take only first 31 layers (index 0 -> 30 relu4-1)
        
        # splitting VGG into 4 seprate Mini-networks 
        enc_layers = list(self.vgg.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]) # ends at relu1-1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]) # ends at relu2-1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # ends at relu3-1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # ends at relu4-1

        # freeze all parameters :- because VGG is not train
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
        
    def forward(self, input, test=False):
        h1 = self.enc_1(input)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)

        # h1,h2,h3 => only used to loss compute
        if test:
            return h4
        return h1, h2, h3, h4
    

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, (3, 3)),
        )

    def forward(self, input):
        return self.decoder(input) 