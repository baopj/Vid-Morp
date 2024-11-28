""" Dataset loader for the Charades-STA dataset """
import os, random, csv, h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext

from . import average_to_fixed_length
from utils.eval import iou
from IPython import embed
import math

def calculate_IOU(groundtruth, predict):
    groundtruth_init = max(0, groundtruth[0])
    groundtruth_end = groundtruth[1]
    predict_init = max(0, predict[0])
    predict_end = predict[1]
    init_min = min(groundtruth_init, predict_init)
    end_max = max(groundtruth_end, predict_end)
    init_max = max(groundtruth_init, predict_init)
    end_min = min(groundtruth_end, predict_end)
    if end_min < init_max:
        return 0
    IOU = (end_min - init_max) * 1.0 / (end_max - init_min)
    return IOU

class CharadesSTA(data.Dataset):

    def __init__(self, args, split='test', similarity_list=None, get_all=False, load_similarity_flag=False, pfga_specific=False, annotations=None):
        super(CharadesSTA, self).__init__()
        self.args = args
        self.feature_type = args['dataset']['feature_type']
        self.data_dir = args['dataset']['root']

        self.durations = {}
        with open(os.path.join(self.data_dir, 'annot', 'Charades_v1_{}.csv'.format(split))) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.durations[row['id']] = float(row['length'])

            anno_file = open(os.path.join(self.data_dir, 'annot', 'charades_sta_{}.txt'.format(split)), 'r')
            annotations = []
            for line in anno_file:
                anno, sent = line.split("##")
                sent = sent.split('.\n')[0]
                vid, s_time, e_time = anno.split(" ")
                s_time = float(s_time)
                e_time = min(float(e_time), self.durations[vid])
                if s_time < e_time:
                    annotations.append(
                        {'video': vid, 'times': [s_time, e_time], 'description': sent, 'duration': self.durations[vid]})
            anno_file.close()
        self.annotations = annotations
        self.epsilon = 1E-10
        # embed()
        self.cfg_anchor = args['model']['config']['simbase']['anchor']
        # embed()
        self.all_anchor_list = self.generate_all_anchor()

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        gt_s_time, gt_e_time = self.annotations[index]['times']
        description = self.annotations[index]['description']
        duration = self.durations[video_id]
        visual_input, visual_mask = self.get_video_features(video_id)
        visual_input = average_to_fixed_length(visual_input, self.args)

        all_anchor_list = self.generate_all_anchor()
        gt_time_feat = [gt_s_time * self.args['dataset']['num_sample_clips'] / duration, gt_e_time * self.args['dataset']['num_sample_clips'] / duration]
        anchor_input = self.generate_anchor_params(all_anchor_list, gt_time_feat)
        anchor_input = np.array(anchor_input)

        item = {
            'sentence': description,
            'gt_boundary': self.annotations[index]['times'],
            'visual_input': visual_input,
            'anno_idx': index,
            'gt_simbase': anchor_input,
            'duration': duration,
            'video_len': self.args['dataset']['num_sample_clips'],
        }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        features = np.load(os.path.join(self.data_dir, 'feat', f'{vid}.npy'))
        features = torch.from_numpy(features).float()
        if self.args['dataset']['normalize']:
            features = F.normalize(features, dim=1)
        vis_mask = torch.ones((features.shape[0], 1))
        return features, vis_mask

    def generate_anchor(self, feat_len,feat_ratio,max_len,output_path): # for 64 as an example
        anchor_list = []
        element_span = max_len / feat_len # 1024/64 = 16
        span_list = []
        for kk in feat_ratio:
            span_list.append(kk * element_span)
        for i in range(feat_len): # 64
            inner_list = []
            for span in span_list:
                left =   i*element_span + (element_span * 1 / 2 - span / 2)
                right =  i*element_span + (element_span * 1 / 2 + span / 2)
                inner_list.append([left,right])
            anchor_list.append(inner_list)
    #     f = open(output_path,'w')
    #     f.write(str(anchor_list))
    #     f.close()
        return anchor_list

    def generate_all_anchor(self):
        all_anchor_list = []
        for i in range(len(self.cfg_anchor['feature_map_len'])):
            anchor_list = self.generate_anchor(self.cfg_anchor['feature_map_len'][i], self.cfg_anchor['scale_ratios_anchor'+str(i+1)],self.args['dataset']['num_sample_clips'],str(i+1)+'.txt')
            all_anchor_list.append(anchor_list)
        return all_anchor_list

    def get_anchor_params_unit(self, anchor,ground_time_step):
        ground_check = ground_time_step[1]-ground_time_step[0]
        if ground_check <= 0:
            return [0.0,0.0,0.0]
        iou = calculate_IOU(ground_time_step,anchor)
        ground_len = ground_time_step[1]-ground_time_step[0]
        ground_center = (ground_time_step[1] - ground_time_step[0]) * 0.5 + ground_time_step[0]
        output_list  = [iou,ground_center,ground_len]
        return output_list


    def generate_anchor_params(self, all_anchor_list,g_position):
        gt_output = np.zeros([len(self.cfg_anchor['feature_map_len']),max(self.cfg_anchor['feature_map_len']),len(self.cfg_anchor['scale_ratios_anchor1'])*3])
        for i in range(len(self.cfg_anchor['feature_map_len'])):
            for j in range(self.cfg_anchor['feature_map_len'][i]):
                for k in range(len(self.cfg_anchor['scale_ratios_anchor1'])):
                    input_anchor = all_anchor_list[i][j][k]
                    output_temp = self.get_anchor_params_unit(input_anchor,g_position)
                    gt_output[i,j,3*k:3*(k+1)]=np.array(output_temp)
        return gt_output