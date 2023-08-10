import warnings

import torch

from build_utils.layers import *
from build_utils.attention_block import *
from Resnest import resnet, res2nest

attention_block = [SE_block, CBAM_block, ECA_block]


class PANet(nn.Module):
    def __init__(self, base_channels=32, base_depth=2, att_tp=2, scale=3):
        super(PANet, self).__init__()
        self.base_channels = base_channels
        self.base_depth = base_depth
        self.att_tp = att_tp
        self.scale = scale
        self.p3_down_channel = Conv(512, 128)
        self.p4_down_channel = Conv(1024, 256)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3 = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1 = C3Res2(base_channels * 16, base_channels * 8, scale=self.scale)

        self.conv_for_feat2 = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2 = C3Res2(base_channels * 8, base_channels * 4, scale=self.scale)

        self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1 = C3Res2(base_channels * 8, base_channels * 8, scale=self.scale)

        self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2 = C3Res2(base_channels * 16, base_channels * 16, scale=self.scale)

        self.att_dc_p5 = attention_block[self.att_tp](256)
        self.att_dc_p4 = attention_block[self.att_tp](128)
        self.att_ds_p3_1 = attention_block[self.att_tp](128)
        self.att_ds_p4_1 = attention_block[self.att_tp](256)

    def forward(self, inputs):
        feat1, feat2, feat3 = inputs

        feat1 = self.p3_down_channel(feat1)
        feat2 = self.p4_down_channel(feat2)

        P5 = self.conv_for_feat3(feat3)

        # P5_upsample = self.att_dc_p5(P5)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.conv3_for_upsample1(P4)
        # 40, 40, 512 -> 40, 40, 256
        P4 = self.conv_for_feat2(P4)

        # P4_upsample = self.att_dc_p4(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3 = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3 = self.conv3_for_upsample2(P3)

        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # P3_downsample = self.att_ds_p3_1(P3_downsample)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # P4_downsample = self.att_ds_p3_1(P4_downsample)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        return P3, P4, P5


class YOLOLayer(nn.Module):
    """
    对YOLO的输出进行处理
    """

    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度
        self.anchor_vec = self.anchors / self.stride
        # batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            yv, xv = torch.meshgrid(torch.arange(self.ny, device=device),
                                    torch.arange(self.nx, device=device), indexing='ij')
            # batch_size, na, grid_h, grid_w, wh
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
            self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    def __init__(self, num_classes=20, img_size=(640, 640), verbose=False):
        super(Darknet, self).__init__()
        self.nc = num_classes
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 特征提取网络
        self.backbone = resnet.resnest50()
        self.conv1x1_feat3 = Conv(2048, 1024)

        # body部分
        self.spp = SPPF(1024, 512)

        self.neck = PANet()


        # 预测头
        self.conv3 = nn.Conv2d(in_channels=128,
                               out_channels=(self.nc + 5) * 3,
                               kernel_size=1,
                               padding=0,
                               bias=False)
        self.conv4 = nn.Conv2d(in_channels=256,
                               out_channels=(self.nc + 5) * 3,
                               kernel_size=1,
                               padding=0,
                               bias=False)
        self.conv5 = nn.Conv2d(in_channels=512,
                               out_channels=(self.nc + 5) * 3,
                               kernel_size=1,
                               padding=0,
                               bias=False)

        self.yolo_layer0 = YOLOLayer(anchors=[[13, 14], [17, 20], [33, 66]],  # anchor list
                                     nc=self.nc,  # number of classes
                                     img_size=img_size,
                                     stride=8)

        self.yolo_layer1 = YOLOLayer(anchors=[[139, 38], [67, 87], [225, 146]],  # anchor list
                                     nc=self.nc,  # number of classes
                                     img_size=img_size,
                                     stride=16)

        self.yolo_layer2 = YOLOLayer(anchors=[[490, 123], [456, 177], [478, 234]],  # anchor list
                                     nc=self.nc,  # number of classes
                                     img_size=img_size,
                                     stride=32)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose)  # print model description

    def forward(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out = []
        if verbose:
            print('0', x.shape)
            str = ""

        _, out1, out2, out3 = self.backbone(x)
        out3 = self.conv1x1_feat3(out3)
        out3_spp = self.spp(out3)
        out = (out1, out2, out3_spp)
        features = self.neck(out)
        yolo_layers = [self.yolo_layer0, self.yolo_layer1, self.yolo_layer2]
        predict_layers = [self.conv3, self.conv4, self.conv5]
        predict_features = []
        for feature, predict_layer in zip(features, predict_layers):
            predict_features.append(predict_layer(feature))
        for predict_feature, yolo_layer in zip(predict_features, yolo_layers):
            yolo_out.append(yolo_layer(predict_feature))

        if self.training:  # train
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)
