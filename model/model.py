import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch

from model.unet import UNet


class UNet_old(BaseModel):
    def __init__(self, n_class):
        super(UNet_old, self).__init__()
        self.n_class = n_class
        # Encoder
        # input: 256x256x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 254x254x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 252x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 126x284x64

        # input: 126x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 124x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 122x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 61x140x128

        # input: 61x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 58x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 56x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 28x68x256

        # input: 28x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 26x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 24x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 12x32x512

        # input: 12x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 10x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 8x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        self.relu = nn.functional.relu

    def forward(self, x):
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.relu(self.e41(xp3))
        xe42 = self.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.relu(self.e51(xp4))
        xe52 = self.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd42 = self.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


class Pix2PixModel(BaseModel):
    def __init__(self, gf_dim, df_dim, c_dim):
        super(Pix2PixModel, self).__init__()
        self.G = Unet(c_dim, gf_dim)
        self.D = Pix2PixDiscriminator(df_dim, c_dim)

    def forward(self, x):
        return self.G.forward(x)


class MiniPix2PixModel(BaseModel):
    def __init__(self, gf_dim, df_dim, c_dim):
        super(MiniPix2PixModel, self).__init__()
        self.G = MiniPix2PixGenerator(gf_dim, c_dim)
        self.D = MiniPix2PixDiscriminator(df_dim, c_dim)

    def forward(self, x):
        return self.G.forward(x)


class MiniImgPix2PixModel(BaseModel):
    def __init__(self, gf_dim, df_dim, c_dim):
        super(MiniImgPix2PixModel, self).__init__()
        self.G = MiniPix2PixGenerator(gf_dim, c_dim, 3)
        self.D = MiniPix2PixDiscriminator(df_dim, c_dim, 3)

    def forward(self, x):
        return self.G.forward(x)


class Pix2PixGenerator(BaseModel):
    def __init__(self, gf_dim, c_dim, num_classes=10):
        super(Pix2PixGenerator, self).__init__()
        self.e1 = cnn_block(c_dim, gf_dim, 4, 2, 1, first_layer=True)
        self.e2 = cnn_block(gf_dim, gf_dim * 2, 4, 2, 1, )
        self.e3 = cnn_block(gf_dim * 2, gf_dim * 4, 4, 2, 1, )
        self.e4 = cnn_block(gf_dim * 4, gf_dim * 8, 4, 2, 1, )
        self.e5 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1, )
        self.e6 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1, )
        self.e7 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1, )
        self.e8 = cnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1, first_layer=True)

        self.d1 = tcnn_block(gf_dim * 8, gf_dim * 8, 4, 2, 1)
        self.d2 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d3 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d4 = tcnn_block(gf_dim * 8 * 2, gf_dim * 8, 4, 2, 1)
        self.d5 = tcnn_block(gf_dim * 8 * 2, gf_dim * 4, 4, 2, 1)
        self.d6 = tcnn_block(gf_dim * 4 * 2, gf_dim * 2, 4, 2, 1)
        self.d7 = tcnn_block(gf_dim * 2 * 2, gf_dim * 1, 4, 2, 1)
        self.d8 = tcnn_block(gf_dim * 1 * 2, num_classes, 4, 2, 1, first_layer=True)  # 256x256
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        e3 = self.e3(F.leaky_relu(e2, 0.2))
        e4 = self.e4(F.leaky_relu(e3, 0.2))
        e5 = self.e5(F.leaky_relu(e4, 0.2))
        e6 = self.e6(F.leaky_relu(e5, 0.2))
        e7 = self.e7(F.leaky_relu(e6, 0.2))
        e8 = self.e8(F.leaky_relu(e7, 0.2))
        d1 = torch.cat([F.dropout(self.d1(F.relu(e8)), 0.5, training=True), e7], 1)
        d2 = torch.cat([F.dropout(self.d2(F.relu(d1)), 0.5, training=True), e6], 1)
        d3 = torch.cat([F.dropout(self.d3(F.relu(d2)), 0.5, training=True), e5], 1)
        d4 = torch.cat([self.d4(F.relu(d3)), e4], 1)
        d5 = torch.cat([self.d5(F.relu(d4)), e3], 1)
        d6 = torch.cat([self.d6(F.relu(d5)), e2], 1)
        d7 = torch.cat([self.d7(F.relu(d6)), e1], 1)
        d8 = self.d8(F.relu(d7))
        o = self.softmax(d8)
        return o

        # return self.tanh(d8)


class Pix2PixDiscriminator(BaseModel):
    def __init__(self, df_dim, c_dim, num_classes=10):
        super(Pix2PixDiscriminator, self).__init__()
        self.conv1 = cnn_block(c_dim + num_classes, df_dim, 4, 1, 1, first_layer=True, maxpool=True)  # 128x128
        self.conv2 = cnn_block(df_dim, df_dim * 2, 4, 1, 1, maxpool=True)  # 64x64
        self.conv3 = cnn_block(df_dim * 2, df_dim * 4, 4, 1, 1, maxpool=True)  # 32 x 32
        self.conv4 = cnn_block(df_dim * 4, df_dim * 8, 4, 1, 1, maxpool=True)  # 16 x 16
        self.conv5 = cnn_block(df_dim * 8, 1, 4, 1, 1, first_layer=True, maxpool=True)  # 7 x 7

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        O = torch.cat([x, y], dim=1)
        O = F.leaky_relu(self.conv1(O), 0.2)
        O = F.leaky_relu(self.conv2(O), 0.2)
        O = F.leaky_relu(self.conv3(O), 0.2)
        O = F.leaky_relu(self.conv4(O), 0.2)
        O = self.conv5(O)
        return self.sigmoid(O)


class MiniPix2PixGenerator(BaseModel):
    def __init__(self, gf_dim, c_dim, num_classes=10):
        super(MiniPix2PixGenerator, self).__init__()
        self.e1 = cnn_block(c_dim, gf_dim, 4, 2, 1, first_layer=True)
        self.e2 = cnn_block(gf_dim, gf_dim * 2, 4, 2, 1, )
        self.e3 = cnn_block(gf_dim * 2, gf_dim * 4, 4, 2, 1, )
        self.e4 = cnn_block(gf_dim * 4, gf_dim * 8, 4, 2, 1, first_layer=True)

        self.d1 = tcnn_block(gf_dim * 8, gf_dim * 4, 4, 2, 1)
        self.d2 = tcnn_block(gf_dim * 4 * 2, gf_dim * 2, 4, 2, 1)
        self.d3 = tcnn_block(gf_dim * 2 * 2, gf_dim * 1, 4, 2, 1)
        self.d4 = tcnn_block(gf_dim * 1 * 2, num_classes, 4, 2, 1, first_layer=True)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        e3 = self.e3(F.leaky_relu(e2, 0.2))
        e4 = self.e4(F.leaky_relu(e3, 0.2))
        d1 = torch.cat([F.dropout(self.d1(F.relu(e4)), 0.5, training=True), e3], 1)
        d2 = torch.cat([F.dropout(self.d2(F.relu(d1)), 0.5, training=True), e2], 1)
        d3 = torch.cat([F.dropout(self.d3(F.relu(d2)), 0.5, training=True), e1], 1)
        d4 = self.d4(F.relu(d3))
        o = self.softmax(d4)
        return o


class MiniPix2PixDiscriminator(BaseModel):
    def __init__(self, df_dim, c_dim, num_classes=10):
        super(MiniPix2PixDiscriminator, self).__init__()
        self.conv1 = cnn_block(c_dim + num_classes, df_dim, 4, 1, 1, first_layer=True, maxpool=True)  # 128x128
        self.conv2 = cnn_block(df_dim, df_dim * 2, 4, 1, 1, maxpool=True)  # 64x64
        self.conv3 = cnn_block(df_dim * 2, df_dim * 4, 4, 1, 1, maxpool=True)  # 32 x 32
        self.conv4 = cnn_block(df_dim * 4, df_dim * 8, 4, 1, 1, maxpool=True)  # 16 x 16
        self.conv5 = cnn_block(df_dim * 8, 1, 4, 1, 1, first_layer=True, maxpool=True)  # 7 x 7

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        O = torch.cat([x, y], dim=1)
        O = F.leaky_relu(self.conv1(O), 0.2)
        O = F.leaky_relu(self.conv2(O), 0.2)
        O = F.leaky_relu(self.conv3(O), 0.2)
        O = F.leaky_relu(self.conv4(O), 0.2)
        O = self.conv5(O)
        return self.sigmoid(O)


class LabelImageGenarator(BaseModel):
    def __init__(self, h_dim, in_dim, out_dim, num_classes=10):
        super(LabelImageGenarator, self).__init__()
        self.num_box = 8

        # Encoder
        self.e1 = cnn_block(in_dim, h_dim, 5, 2, 2)  # 128x128   C: 3 -> 8
        self.e2 = cnn_block(h_dim, h_dim * 2, 5, 2, 2, )  # 64x64   C: 8 -> 16
        self.e3 = cnn_block(h_dim * 2, h_dim * 4, 5, 2, 2, )  # 32 x 32  C: 16 -> 32
        self.e4 = cnn_block(h_dim * 4, h_dim * 8, 5, 2, 2, first_layer=True)  # 16 x 16 C: 32 -> 64

        # Decoder
        self.d1 = tcnn_block(h_dim * 8, h_dim * 4, 5, 2, 2, 1)  # 32 x 32 C: 64 -> 32
        self.d2 = tcnn_block(h_dim * (4 + 4), h_dim * 4, 5, 2, 2, 1)  # 64x64  C: 32 + 32 -> 32
        self.d3 = tcnn_block(h_dim * (4 + 2), h_dim * 2, 5, 2, 2, 1)  # 128x128 C: 32 + 16 -> 16
        self.d4 = tcnn_block(h_dim * (2 + 1), out_dim, 5, 2, 2, 1)  # 256x256 C: 16 + 8 -> out

        # Creater
        c_dim = self.num_box + out_dim
        self.c1 = cnn_block(c_dim, c_dim * 2, 5, 1, 2)
        self.c2 = cnn_block(c_dim * 2, c_dim * 4, 5, 1, 2)
        self.c3 = cnn_block(c_dim * 4, num_classes, 9, 1, 4)

        self.creater = nn.Sequential(
            self.c1,
            nn.ReLU(),
            self.c2,
            nn.ReLU(),
            self.c3,
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def boundingImages(self, boundingPoints):
        # bound box (Batch, 8*4)
        image = torch.zeros(boundingPoints.shape[0], self.num_box, 256, 256)
        boundingPoints = boundingPoints.view(boundingPoints.shape[0], self.num_box, 4)
        for i in range(boundingPoints.shape[0]):
            for box in range(self.num_box):
                target = boundingPoints[i, box]
                image[i, box, int(target[1]):int(target[1] + target[3]), int(target[0]):int(target[0] + target[2])] = 1
        return image

    def forward(self, x, box):

        e1 = self.e1(x)
        e2 = self.e2(F.leaky_relu(e1, 0.2))
        e3 = self.e3(F.leaky_relu(e2, 0.2))
        e4 = self.e4(F.leaky_relu(e3, 0.2))

        d1 = torch.cat([self.d1(F.relu(e4)), e3], 1)
        d2 = torch.cat([self.d2(F.relu(d1)), e2], 1)
        d3 = torch.cat([self.d3(F.relu(d2)), e1], 1)
        d4 = self.d4(F.relu(d3))

        box = self.boundingImages(box)
        x = torch.cat((d4, box), 1)

        x = self.creater(x)
        o = self.softmax(x)
        print(o[0, :, 74, 160])
        return o


class Unet(UNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class BoundingBoxPredicter(BaseModel):
    def __init__(self, in_channel, cnn_channel, out_channel=4, num_classes=8):
        super(BoundingBoxPredicter, self).__init__()
        # in 3x256x256
        self.e1 = cnn_block(in_channel, cnn_channel, 4, padding=2, maxpool=True, first_layer=True)
        # out nx128x128
        self.e2 = cnn_block(cnn_channel, cnn_channel * 2, 4, padding=2, maxpool=True)
        # out 2nx64x64
        self.e3 = cnn_block(cnn_channel * 2, cnn_channel * 4, 4, padding=2, maxpool=True)
        # out 4n32x32
        self.e4 = cnn_block(cnn_channel * 4, out_channel, 4, padding=2, maxpool=True, first_layer=True)
        # out OCx16x16
        self.linear = nn.Linear(out_channel * 16 * 16,
                                num_classes * 4)  # OCx16x16 -> 4*num_classes (x,y,w,h)  (1024 => 32)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(F.leaky_relu(x, 0.2))
        x = self.e3(F.leaky_relu(x, 0.2))
        x = self.e4(F.leaky_relu(x, 0.2))
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        o = self.relu(x)
        return o


def cnn_block(in_channels, out_channels, kernel_size, stride=1, padding=0, first_layer=False, maxpool=False):
    if first_layer:
        if maxpool:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.MaxPool2d(2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            )
    else:
        if maxpool:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
            )


def tcnn_block(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, first_layer=False):
    if first_layer:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                  output_padding=output_padding)

    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               output_padding=output_padding),
            nn.BatchNorm2d(out_channels, momentum=0.1, eps=1e-5),
        )
