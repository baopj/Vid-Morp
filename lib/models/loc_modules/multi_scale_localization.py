import torch
import torch.nn as nn
from IPython import embed

class BNReLUConv(nn.Module):
    def __init__(self, kernel_size, stride, input_channels=128, out_channels=128):
        super(BNReLUConv, self).__init__()
        self.bn = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(input_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class MultiScaleLocalization(nn.Module):
    def __init__(self, cfg):
        super(MultiScaleLocalization, self).__init__()
        self.cfg_multi_scale_loc = cfg['simbase']['multi_scale_loc']
        pred_input_size = cfg['simbase']['pred']['input_size']
        pred_reg_dim = cfg['simbase']['pred']['reg_dim']
        cfg_anchor = cfg['simbase']['anchor']
        self.cfg_anchor = cfg['simbase']['anchor']

        self.conv1 = nn.Conv1d(cfg['textual_converter']['reduction']['output_size'], pred_input_size, kernel_size=self.cfg_multi_scale_loc['conv1_kernel_size'], stride=1, padding=cfg['simbase']['multi_scale_loc']['conv1_kernel_size']//2)
        self.bn_relu_conv = BNReLUConv(kernel_size=self.cfg_multi_scale_loc['conv1_kernel_size'], stride=1)
        self.pool = nn.MaxPool1d(2)

        conv_pool_layers = []
        for i in range(1, len(self.cfg_anchor['feature_map_len'])+1):
            conv_pool_layer = BNReLUConv(kernel_size=3, stride=2)
            setattr(self, f'conv_pool{i}', conv_pool_layer)
            conv_pool_layers.append(conv_pool_layer)
        self.conv_pool_layers = conv_pool_layers

        overlap_layers_a = []
        overlap_layers_b = []
        reg_layers_a = []
        reg_layers_b = []

        for anchor_id in range(1, len(self.cfg_anchor['feature_map_len'])+1):
            overlap_layer_a = BNReLUConv(kernel_size=self.cfg_multi_scale_loc['overlap_a_kernel_size'], stride=1,
                                         input_channels=pred_input_size,
                                         out_channels=pred_input_size)
            overlap_layer_b = BNReLUConv(kernel_size=self.cfg_multi_scale_loc['overlap_b_kernel_size'], stride=1,
                                         input_channels=pred_input_size,
                                         out_channels=len(cfg_anchor['scale_ratios_anchor%d' % anchor_id]))
            reg_layer_a = BNReLUConv(kernel_size=self.cfg_multi_scale_loc['reg_a_kernel_size'], stride=1,
                                     input_channels=pred_input_size,
                                     out_channels=pred_input_size)
            reg_layer_b = BNReLUConv(kernel_size=self.cfg_multi_scale_loc['reg_b_kernel_size'], stride=1,
                                     input_channels=pred_input_size,
                                     out_channels=len(cfg_anchor['scale_ratios_anchor%d' % anchor_id]) * pred_reg_dim)

            overlap_layers_a.append(overlap_layer_a)
            overlap_layers_b.append(overlap_layer_b)
            reg_layers_a.append(reg_layer_a)
            reg_layers_b.append(reg_layer_b)

            setattr(self, f'overlap_layer_{anchor_id}_a', overlap_layer_a)
            setattr(self, f'overlap_layer_{anchor_id}_b', overlap_layer_b)
            setattr(self, f'reg_layer_{anchor_id}_a', reg_layer_a)
            setattr(self, f'reg_layer_{anchor_id}_b', reg_layer_b)

        self.overlap_layers_a = overlap_layers_a
        self.overlap_layers_b = overlap_layers_b
        self.reg_layers_a = reg_layers_a
        self.reg_layers_b = reg_layers_b

    def forward(self, fusion_input):
        # fusion_input (b,128,64)
        x = self.conv1(fusion_input)  # x (b,128,64)
        x = self.bn_relu_conv(x)  # x (b,128,64)
        x = self.pool(x)  # x (b,128,32)

        anchors = []
        current_input = x
        for i in range(1, len(self.cfg_anchor['feature_map_len'])+1):
            current_input = getattr(self, f'conv_pool{i}')(current_input)
            anchors.append(current_input)
        predict_overlaps = []
        predict_regs = []

        for i in range(len(self.cfg_anchor['feature_map_len'])):
            anchor = anchors[i]
            overlap_layer_a = self.overlap_layers_a[i]
            overlap_layer_b = self.overlap_layers_b[i]
            reg_layer_a = self.reg_layers_a[i]
            reg_layer_b = self.reg_layers_b[i]
            predict_overlap = overlap_layer_a(anchor)
            predict_overlap = overlap_layer_b(predict_overlap)
            predict_reg = reg_layer_a(anchor)
            predict_reg = reg_layer_b(predict_reg)
            predict_reg = torch.tanh(predict_reg)
            predict_overlaps.append(predict_overlap)
            predict_regs.append(predict_reg)
        return predict_overlaps, predict_regs
