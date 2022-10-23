from .aspp import *
from functools import partial
from generalframeworks.networks.uncer_head import Uncertainty_head_conv
##### For PRCL_Loss #####
class DeepLabv3Plus(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=21, output_dim=256):
        super(DeepLabv3Plus, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        #self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change the num of input channel
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))

        return prediction, representation

class DeepLabv3Plus_multi_rep(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=21, output_dim=256):
        super(DeepLabv3Plus_multi_rep, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        #self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change the num of input channel
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

        self.representation_1 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))
        representation_1 = self.representation_1(torch.cat([x_low, output_feature], dim=1))


        return prediction, representation, representation_1

##### For PRCL Loss #####
class DeepLabv3Plus_with_un(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=21, output_dim=256):
        super(DeepLabv3Plus_with_un, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        #self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change the num of input channel
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))
    
        
        return prediction, representation, torch.cat([x_low, output_feature], dim=1)


class DeepLabv3Plus_nr(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=21, output_dim=256):
        super(DeepLabv3Plus_nr, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        #self.resnet_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Change the num of input channel
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))

        return prediction

class DeepLabv3Plus_UncerHead(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, num_classes=21, output_dim=256):
        super(DeepLabv3Plus_UncerHead, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]

        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

        self.uncer_head = Uncertainty_head_conv()

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)
        prediction = self.classifier(torch.cat([x_low, output_feature], dim=1))
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))

        uncer_log = self.uncer_head(torch.cat([x_low, output_feature], dim=1))

        return prediction, representation, uncer_log

class DeepLabv3Plus_light(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=16, output_dim=256, apex_sync_bn=False):
        super(DeepLabv3Plus_light, self).__init__()
        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
            aspp_dilate = [12, 24, 36]

        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
            aspp_dilate = [6, 12, 18]
        if apex_sync_bn:
            self.norm = nn.BatchNorm2d
        else:
            self.norm = nn.BatchNorm2d
        # take pre-defined ResNet, except AvgPool and FC
        self.resnet_conv1 = orig_resnet.conv1
        self.resnet_bn1 = orig_resnet.bn1
        self.resnet_relu1 = orig_resnet.relu
        self.resnet_maxpool = orig_resnet.maxpool

        self.resnet_layer1 = orig_resnet.layer1
        self.resnet_layer2 = orig_resnet.layer2
        self.resnet_layer3 = orig_resnet.layer3
        self.resnet_layer4 = orig_resnet.layer4

        self.ASPP = ASPP(2048, aspp_dilate, self.norm)

        self.project = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            self.norm(48),
            nn.ReLU(inplace=True),
        )

        self.representation = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            self.norm(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)

            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        h_w = x.shape[2:]
        # with ResNet-50 Encoder
        x = self.resnet_relu1(self.resnet_bn1(self.resnet_conv1(x)))
        x = self.resnet_maxpool(x)

        x_low = self.resnet_layer1(x)
        x = self.resnet_layer2(x_low)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        feature = self.ASPP(x)

        # Decoder
        x_low = self.project(x_low)
        output_feature = F.interpolate(feature, size=x_low.shape[2:], mode='bilinear', align_corners=True)

        raw_feature = torch.cat([x_low, output_feature], dim=1)
        representation = self.representation(torch.cat([x_low, output_feature], dim=1))
        
        return representation, raw_feature