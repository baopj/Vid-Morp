import torch
import torch.nn as nn
import numpy as np

def collate_fn(batch):
    batch_anno_idxs = [b['anno_idx'] for b in batch]
    batch_vis_feats = [b['visual_input'] for b in batch]
    batch_duration = [b['duration'] for b in batch]
    batch_sentence = [b['sentence'] for b in batch]
    batch_video_len = [b['video_len'] for b in batch]
    batch_gt_boundary = [b['gt_boundary'] for b in batch]
    batch_gt_simbase = [b['gt_simbase'] for b in batch]

    batch_data = {
        'annot_idx': batch_anno_idxs,
        'duration': batch_duration,
        'gt_time': batch_gt_boundary,
        'sentence': batch_sentence,
        'net_input':{
            'gt_simbase': torch.from_numpy(np.array(batch_gt_simbase)).float(),
            'videoFeat': nn.utils.rnn.pad_sequence(batch_vis_feats, batch_first=True).float(),
            'videoLen': torch.from_numpy(np.array(batch_video_len)).long(),
        }
    }

    return batch_data

def average_to_fixed_length(visual_input, args):
    num_sample_clips = args['dataset']['num_sample_clips']
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

from datasets.charades import CharadesSTA
