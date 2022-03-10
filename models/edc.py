from models.misc.gridnet import MultiInputGridNet
from models.misc import resnet_3D
from models.misc.resnet_3D import BasicStem, Conv3DSimple
import torch
import torch.nn as nn
import sys
from torch.nn import functional as F
from models.misc import Identity
import cupy_module.adacof as adacof


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class EDC(nn.Module):
    def __init__(self, args):
        super(EDC, self).__init__()
        self.kernel_size = args.kernel_size
        self.dilation = args.dilation
        
        batchnorm = Identity
        resnet_3D.batchnorm = Identity
        class Encoder(nn.Module):
            def __init__(self, block, conv_makers, layers,
                        stem, channels=[16,32,64,96,128,160]):
                """Generic resnet video generator.

                Args:
                    block (nn.Module): resnet building block
                    conv_makers (list(functions)): generator function for each layer
                    layers (List[int]): number of blocks per layer
                    stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
                    zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
                """
                super(Encoder, self).__init__()
                self.inplanes = channels[0] # output channel of first stem

                self.stem = stem(channels[0])

                self.layer1 = self._make_layer(block, conv_makers[0], channels[1], layers[0], stride=2, temporal_stride=1)
                self.layer2 = self._make_layer(block, conv_makers[1], channels[2], layers[1], stride=2 , temporal_stride=1)
                self.layer3 = self._make_layer(block, conv_makers[2], channels[3], layers[2], stride=2 , temporal_stride=1)
                self.layer4 = self._make_layer(block, conv_makers[3], channels[4], layers[3], stride=2, temporal_stride=1)
                self.layer5 = self._make_layer(block, conv_makers[4], channels[5], layers[4], stride=2, temporal_stride=1)

                # init weights
                self._initialize_weights()

            def forward(self, x):
                tensorConv0 = self.stem(x)
                tensorConv1 = self.layer1(tensorConv0)
                tensorConv2 = self.layer2(tensorConv1)
                tensorConv3 = self.layer3(tensorConv2)
                tensorConv4 = self.layer4(tensorConv3)
                tensorConv5 = self.layer5(tensorConv4)
                return tensorConv0, tensorConv1, tensorConv2, tensorConv3, tensorConv4, tensorConv5

            def _make_layer(self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None):
                downsample = None

                if stride != 1 or self.inplanes != planes * block.expansion:
                    ds_stride = conv_builder.get_downsample_stride(stride , temporal_stride)
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=ds_stride, bias=False),
                        batchnorm(planes * block.expansion)
                    )
                    stride = ds_stride

                layers = []
                layers.append(block(self.inplanes, planes, conv_builder, stride, downsample ))

                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes, conv_builder ))

                return nn.Sequential(*layers)

            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Conv3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                                nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm3d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)

        class Decoder(nn.Module):
            def __init__(self, intLevel, kernel_size, dilation, channels=[3,16,32,64,96,128,160]):
                super(Decoder, self).__init__()
                self.intLevel = intLevel
                self.kernel_size = kernel_size
                self.dilation = dilation
                self.kernel_pad = int(((kernel_size - 1) * dilation) / 2.0)
                self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

                intCurrent = channels[intLevel]
                depth = 4 #if intLevel == 6 else 5
                if intLevel < 6:
                    # prevOut -> curFeat
                    # turn b,8,4,4 into b,8,8,8
                    # turn b,8,8,8 into b,8,16,16
                    self.netUpOut = nn.UpsamplingBilinear2d(scale_factor=2)

                    # prevFeat -> curFeat
                    # turn b,intPreviou,depth,4,4 into b,intCurrent/2,1,8,8
                    # turn b,intPreviou,depth,8,8 into b,intCurrent/2,1,16,16
                    self.netUpFeat = nn.Sequential(
                        nn.Conv2d(64, intCurrent, kernel_size=(3,3), stride=1, padding=1),
                        nn.ConvTranspose2d(intCurrent, intCurrent//2, kernel_size=(4,4), stride=2, padding=1)
                    )

                # turn b,160,4,4,4   into b,8,1,4,4
                # turn b,128,4,8,8   into b,8,1,8,8
                # turn b,96,4,16,16  into b,8,1,16,16
                self.netOne = nn.Sequential(
                    nn.Conv3d(intCurrent, intCurrent//2, kernel_size=3, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv3d(intCurrent//2, intCurrent//2, kernel_size=(depth,3,3), padding=(0,1,1), bias=False),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )
                cin = [318, intCurrent+12+300+4, intCurrent+12+300+4, intCurrent, intCurrent, intCurrent, 80][intLevel]
                self.netTwo = nn.Sequential(
                    nn.Conv2d(cin, 128, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
                if intLevel < 4:
                    self.netAlpha0 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netAlpha1 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netAlpha2 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netAlpha3 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netBeta0 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netBeta1 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netBeta2 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netBeta3 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                    )

                    self.netWeight0 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                        nn.Softmax(dim=1)
                    )

                    self.netWeight1 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                        nn.Softmax(dim=1)
                    )

                    self.netWeight2 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                        nn.Softmax(dim=1)
                    )

                    self.netWeight3 = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1),
                        nn.Softmax(dim=1)
                    )

                    self.occconv = nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(inplace=False, negative_slope=0.1),
                        nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
                        nn.Softmax(dim=1)
                    )

            def forward(self, tenFeat, objPrevious, images):
                """
                args:
                tenFeat: (B,intCurrent,D,H,W) feature map at current level
                tenOut: (B,8,H/2,W/2)
                images: (B,3,4,H/2,W/2)
                """
                if objPrevious is None:
                    tenFeat = self.netOne(tenFeat).squeeze(2)
                
                elif objPrevious is not None and self.intLevel >= 3:
                    tenFeatPrev = self.netUpFeat(objPrevious['tenFeat'])
                    tenFeat = torch.cat([self.netOne(tenFeat).squeeze(2), tenFeatPrev], 1)

                elif objPrevious is not None and self.intLevel < 3:
                    # downsample input images to current scale
                    h,w = images.shape[3:]
                    images = torch.cat(torch.unbind(images,2),dim=1)
                    images = F.interpolate(images, size=(h//2**self.intLevel, w//2**self.intLevel), mode='bilinear', align_corners=False)
                    # upsample flows
                    tenAlpha = self.netUpOut(objPrevious['tenAlpha']) * 2.
                    tenBeta = self.netUpOut(objPrevious['tenBeta']) * 2.
                    tenWeight = self.netUpOut(objPrevious['tenWeight'])
                    tenOcc = self.netUpOut(objPrevious['tenOcc'])
                    tenFeatPrev = self.netUpFeat(objPrevious['tenFeat'])
                    # warp inputs
                    ks = self.kernel_size**2
                    I1_warped = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,:3]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
                    I2_warped = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,3:6]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
                    I3_warped = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,6:9]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
                    I4_warped = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,9:]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
                    tenPrev = torch.cat([tenFeatPrev, # (B,intCurrent/2,H,W) 
                                         tenOcc, # (B,4,H,W)
                                         tenAlpha,  # (B,100,H,W)
                                         tenBeta, # (B,100,H,W)
                                         tenWeight, # (B,100,H,W)
                                         I1_warped, I2_warped, I3_warped, I4_warped], 1) # (B,12,H,W)
                    tenFeat = torch.cat([self.netOne(tenFeat).squeeze(2), tenPrev], dim=1)

                tenFeat = self.netTwo(tenFeat)
                if self.intLevel < 4:
                    return {'tenAlpha': torch.cat([self.netAlpha0(tenFeat), self.netAlpha1(tenFeat), self.netAlpha2(tenFeat), self.netAlpha3(tenFeat)], dim=1),
                            'tenBeta': torch.cat([self.netBeta0(tenFeat), self.netBeta1(tenFeat), self.netBeta2(tenFeat), self.netBeta3(tenFeat)], dim=1),
                            'tenWeight': torch.cat([self.netWeight0(tenFeat), self.netWeight1(tenFeat), self.netWeight2(tenFeat), self.netWeight3(tenFeat)], dim=1),
                            'tenOcc': self.occconv(tenFeat),
                            'tenFeat': tenFeat}
                return {'tenFeat': tenFeat}

        self.kernel_pad = int(((self.kernel_size - 1) * self.dilation) / 2.0)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.netExtractor = Encoder(BasicBlock, conv_makers=[Conv3DSimple]*5, layers=[2,2,2,2,2], stem=BasicStem, channels=[16,32,64,96,128,160])
        self.netZer = Decoder(0, self.kernel_size, self.dilation)
        self.netOne = Decoder(1, self.kernel_size, self.dilation)
        self.netTwo = Decoder(2, self.kernel_size, self.dilation)
        self.netThr = Decoder(3, self.kernel_size, self.dilation)
        self.netFou = Decoder(4, self.kernel_size, self.dilation)
        self.netFiv = Decoder(5, self.kernel_size, self.dilation)
        self.netSix = Decoder(6, self.kernel_size, self.dilation)
        self.synthesis_net = MultiInputGridNet(in_chs=(3*5+64*4, 3, 3), out_chs=3, grid_chs=(32,64,96))
        self.context = nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3), bias=False)

    def forward(self, I0, I1, I2, I3, *args):
        h0 = int(list(I1.size())[2])
        w0 = int(list(I1.size())[3])
        h2 = int(list(I2.size())[2])
        w2 = int(list(I2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 128 != 0:
            pad_h = 128 - (h0 % 128)
            I0 = F.pad(I0, (0, 0, 0, pad_h), mode='reflect')
            I1 = F.pad(I1, (0, 0, 0, pad_h), mode='reflect')
            I2 = F.pad(I2, (0, 0, 0, pad_h), mode='reflect')
            I3 = F.pad(I3, (0, 0, 0, pad_h), mode='reflect')
            h_padded = True

        if w0 % 128 != 0:
            pad_w = 128 - (w0 % 128)
            I0 = F.pad(I0, (0, pad_w, 0, 0), mode='reflect')
            I1 = F.pad(I1, (0, pad_w, 0, 0), mode='reflect')
            I2 = F.pad(I2, (0, pad_w, 0, 0), mode='reflect')
            I3 = F.pad(I3, (0, pad_w, 0, 0), mode='reflect')
            w_padded = True

        images = torch.stack([I0, I1, I2, I3] , dim=2)
        # Instance normalisation
        mu = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mu 

        It_pyr = []
        feat_pyr = self.netExtractor(images)
        objEstimate = self.netSix(feat_pyr[-1], None, images)
        objEstimate = self.netFiv(feat_pyr[-2], objEstimate, images)
        objEstimate = self.netFou(feat_pyr[-3], objEstimate, images)

        objEstimate = self.netThr(feat_pyr[-4], objEstimate, images)
        It_pyr.append(self.fusion(objEstimate, images))

        objEstimate = self.netTwo(feat_pyr[-5], objEstimate, images)
        It_pyr.append(self.fusion(objEstimate, images))

        objEstimate = self.netOne(feat_pyr[-6], objEstimate, images)
        It_pyr.append(self.fusion(objEstimate, images))

        # final level
        objEstimate = self.netZer(images, objEstimate, images)

        It_0, It_1, It_2, It_3 = self.backwarp(objEstimate, images)
        It_fuse = It_0*objEstimate['tenOcc'][:,:1]+\
                  It_1*objEstimate['tenOcc'][:,1:2]+\
                  It_2*objEstimate['tenOcc'][:,2:3]+\
                  It_3*objEstimate['tenOcc'][:,3:]
        Ct_0, Ct_1, Ct_2, Ct_3 = self.backwarp(objEstimate, torch.stack([self.context(images[:,:,0]),
                                                                         self.context(images[:,:,1]),
                                                                         self.context(images[:,:,2]),
                                                                         self.context(images[:,:,3])], dim=2))
        # final synthesis
        It = self.synthesis_net(torch.cat([It_fuse, It_0, It_1, It_2, It_3, Ct_0, Ct_1, Ct_2, Ct_3], dim=1), It_pyr[-1], It_pyr[-2])

        mu = mu.squeeze(2)
        It += mu

        if h_padded:
            It = It[:, :, 0:h0, :]
        if w_padded:
            It = It[:, :, :, 0:w0]

        if self.training:
            return {'frame1': It}

        return It

    def fusion(self, objEstimate, images):
        tenAlpha = objEstimate['tenAlpha']
        tenBeta = objEstimate['tenBeta']
        tenWeight = objEstimate['tenWeight']
        tenOcc = objEstimate['tenOcc']

        # downsample images
        h,w = tenAlpha.shape[2:]
        c = images.shape[1]
        images = torch.cat(torch.unbind(images,2),dim=1)
        images = F.interpolate(images, size=(h,w), mode='bilinear', align_corners=False)
        ks = self.kernel_size**2
        I0_W  = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,:c]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
        I1_W = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,c:2*c]), tenWeight[:,ks:2*ks].contiguous(), tenAlpha[:,ks:2*ks].contiguous(), tenBeta[:,ks:2*ks].contiguous(), self.dilation)*1.
        I2_W  = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,2*c:3*c]), tenWeight[:,2*ks:3*ks].contiguous(), tenAlpha[:,2*ks:3*ks].contiguous(), tenBeta[:,2*ks:3*ks].contiguous(), self.dilation)*1.
        I3_W = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,3*c:]), tenWeight[:,3*ks:].contiguous(), tenAlpha[:,3*ks:].contiguous(), tenBeta[:,3*ks:].contiguous(), self.dilation)*1.
        It_fuse = I0_W*tenOcc[:,:1]+I1_W*tenOcc[:,1:2]+I2_W*tenOcc[:,2:3]+I3_W*tenOcc[:,3:]

        return It_fuse

    def backwarp(self, objEstimate, images):
        tenAlpha = objEstimate['tenAlpha']
        tenBeta = objEstimate['tenBeta']
        tenWeight = objEstimate['tenWeight']

        # downsample images
        h,w = tenAlpha.shape[2:]
        c = images.shape[1]
        images = torch.cat(torch.unbind(images,2),dim=1)
        images = F.interpolate(images, size=(h,w), mode='bilinear', align_corners=False)
        ks = self.kernel_size**2
        I0_W  = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,:c]), tenWeight[:,:ks].contiguous(), tenAlpha[:,:ks].contiguous(), tenBeta[:,:ks].contiguous(), self.dilation)*1.
        I1_W = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,c:2*c]), tenWeight[:,ks:2*ks].contiguous(), tenAlpha[:,ks:2*ks].contiguous(), tenBeta[:,ks:2*ks].contiguous(), self.dilation)*1.
        I2_W  = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,2*c:3*c]), tenWeight[:,2*ks:3*ks].contiguous(), tenAlpha[:,2*ks:3*ks].contiguous(), tenBeta[:,2*ks:3*ks].contiguous(), self.dilation)*1.
        I3_W = adacof.FunctionAdaCoF.apply(self.modulePad(images[:,3*c:]), tenWeight[:,3*ks:].contiguous(), tenAlpha[:,3*ks:].contiguous(), tenBeta[:,3*ks:].contiguous(), self.dilation)*1.

        return I0_W, I1_W, I2_W, I3_W