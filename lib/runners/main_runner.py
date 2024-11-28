import collections, logging, copy, os, torch, torchtext, pickle, json, time, operator, csv, random, hashlib
import torch.nn.functional as F
import numpy as np
from utils.input_output import TimeMeter, AverageMeter
from pathlib import Path
from tqdm import tqdm
from torch import nn
from terminaltables import AsciiTable
from concurrent.futures import ThreadPoolExecutor, as_completed

from IPython import embed

def info(msg):
    print(msg)

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick
    union = list(map(operator.sub, x2, x1)) # union = x2-x1

    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)
        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

def display_python_performance(x, data='CharadesSTA'):
    i_list = [1]
    j_list = [3, 5, 7]

    display_list = []
    for i in i_list:
        display_itm = []
        for j in j_list:
            key = 'R{}@0.{}'.format(i, j)
            display_itm.append(key)
        display_itm.append('mIoU')
        display_itm.append(f'Avg R{i}')
        display_list.append(display_itm)

    for i in i_list:
        display_itm = []
        for j in j_list:
            key = 'IoU@0.{}'.format(j)
            display_itm.append('{:.2f}'.format(x[key].avg * 100))
        display_itm.append('{:.2f}'.format(x['mIoU'].avg * 100))
        display_itm.append('{:.2f}'.format(get_average_performance(x, data)))
        display_list.append(display_itm)

    table = AsciiTable(display_list)
    return table.table

def print_md_performance(x, data='CharadesSTA'):
    item = ''
    i_list = [1]
    j_list = [3, 5, 7]

    for i in i_list:
        for j in j_list:
            key = 'IoU@0.{}'.format(j)
            item += '| {:.2f} '.format(x[key].avg * 100)
        item += '| {:.2f} |\n'.format(x[f'mIoU'].avg * 100)

    for i in i_list:
        for j in j_list:
            key = 'IoU@0.{}'.format(j)
            item += '&{:.2f} '.format(x[key].avg * 100)
        item += '&{:.2f} |'.format(x[f'mIoU'].avg * 100)
        if i_list[-1] != i:
            item += '\n'
    return item

def get_average_performance(x, data='CharadesSTA', i=1):
    j_list = [3, 5, 7]
    average = 0.0
    for j in j_list:
        key = 'IoU@0.{}'.format(j)
        average += x[key].avg
    average = average / len(j_list)
    average *= 100.0
    return average

def print_python_performance(x, data='CharadesSTA'):
    item = ''
    i_list = [1]
    j_list = [3, 5, 7]

    item += '{}:  {:.2f}\n'.format('average_R1', get_average_performance(x, data))
    item += '{}:        {:.2f}\n'.format('mIoU', x['mIoU'].avg * 100)
    for i in i_list:
        for j in j_list:
            key = 'IoU@0.{}'.format(j)
            item += '{}: {:.2f}  '.format(key, x[key].avg * 100)
    return  item

class ResultSaveObj(object):
    def __init__(self, args):
        self.args = args
        self.save_flag = args['train']['result_saved_flag']
        self.video_info = collections.defaultdict(list)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def add_batch(self, save_result):
        # if not self.save_flag:
        #     return
        for k, v in save_result.items():
            self.video_info[k] += v

    def eval(self):
        gt_time = np.array(self.video_info['gt_time'])
        pred_time = np.array(self.video_info['pred_time'])
        if len(pred_time.shape) == 2:
            pred_time = pred_time[None]
        else:
            pred_time = pred_time.transpose((1, 0, 2))[0:1]
        metric_dict_1 = top_n_metric(pred_time, gt_time)
        # embed()
        return metric_dict_1

class MainRunner:
    def __init__(self, args, similarity_list=None):
        self.process_args(args)
        self._build_dataset()

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

    def process_args(self, args):
        cfg_simbase = args['model']['config']['simbase']
        anchor_ratio_list = np.linspace(cfg_simbase['anchor_meta']['scale_ratios_start'], 1.0,
                                        cfg_simbase['anchor_meta']['scale_ratios_num']).tolist()
        cfg_simbase['anchor']['feature_map_len'] = []
        assert args['dataset']['num_sample_clips'] % 4 == 0
        tmp_num_clips = args['dataset']['num_sample_clips'] // 4
        anchor_id = 0
        while tmp_num_clips > 0:
            anchor_id += 1
            cfg_simbase['anchor'][f"scale_ratios_anchor{anchor_id}"] = copy.deepcopy(anchor_ratio_list)
            cfg_simbase['anchor']['feature_map_len'].append(tmp_num_clips)
            assert tmp_num_clips % 2 == 0 or tmp_num_clips == 1
            tmp_num_clips = tmp_num_clips // 2

        assert len(args['model']['config']['simbase']['anchor']['feature_map_len']) > 0
        args['model']['config']['num_sample_clips'] = args['dataset']['num_sample_clips']
        self.args = args

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()
        self.model.clip_model.eval()

        def print_log():
            msg = 'Epoch {}, Batch {}\n'.format(epoch, bid)
            for k, v in loss_meter.items():
                msg += '\t{} = {:.4f}'.format(k, v.avg)
                v.reset()
            msg += '\n'
            # msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        if self.args['dataset']['name'].startswith('Charades'):
            display_n_batches, bid = 100, 0
        else:
            display_n_batches, bid = 100, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            duration = np.array(batch['duration'])

            ##################################
            # 计算gt
            output_dict = self.model(**net_input, sentence=batch['sentence'])
            loss_dict = output_dict['loss']
            tot_loss = loss_dict['tot_loss']
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.num_updates += 1
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v.item())

            if bid % display_n_batches == 0:
                print_log()
            # break #DEBUG

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, data_loader=None, setting_msg=''):
        if data_loader is None:
            data_loader = self.test_loader
        self.result_save_obj = ResultSaveObj(self.args)
        self.model.eval()
        metrics_logger = collections.defaultdict(lambda: AverageMeter())

        with torch.no_grad():
            bid = 0
            info('2. Evaluating on {}'.format(self.args['dataset']['name']))
            for batch in tqdm(data_loader):
            # for batch in data_loader:
                bid += 1
                net_input = move_to_cuda(batch['net_input'])
                output_dict = self.model(**net_input, sentence=batch['sentence'])

                duration = np.array(batch['duration'])
                video_len = batch['net_input']['videoLen'].detach().cpu().numpy()
                pred_time = self.get_proposal_results(output_dict['pred']['pred_overlap'], output_dict['pred']['pred_reg'], duration)
                save_dict = {}
                save_dict['gt_time'] = batch['gt_time']
                save_dict['pred_time'] = pred_time
                self.result_save_obj.add_batch(save_dict)

        eval_dict = self.result_save_obj.eval()

        for key, v in eval_dict.items():
            metrics_logger[key].update(v, 1)

        info('#' * 60)
        if len(setting_msg) > 0:
            msg = '3. Results on {} ({})'.format(self.args['dataset']['name'], setting_msg)
        else:
            msg = '3. Results on {}'.format(self.args['dataset']['name'])
        info(msg)
        # info('#' * 60)

        msg = display_python_performance(metrics_logger, self.args['dataset']['name'])
        info(msg)
        return metrics_logger

    def _build_dataset(self):
        import pickle
        import datasets as da
        from torch.utils.data import DataLoader
        args = self.args
        cls = getattr(da, args['dataset']['name'], None)
        self.test_set = cls(args=args, split='test')
        # info('test: {} samples'.format(len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=da.collate_fn,
                                      num_workers=1)

    def _build_model(self):
        model_config = self.args['model']
        import models
        self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.float().cuda()

    def _build_optimizer(self):
        model = self.model
        cfg = self.args['train']['optimizer']
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg['lr']
            weight_decay = cfg['weight_decay']
            if "bias" in key:
                lr = cfg['lr'] * cfg['lr_factor']
                weight_decay = cfg['weight_decay_bias']
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg['name'] == "SGD":
            self.optimizer = torch.optim.SGD(params, cfg['lr'], momentum=cfg['momentum'],
                                        weight_decay=cfg['weight_decay'])
        elif cfg['name'] == "Adam":
            self.optimizer = torch.optim.Adam(params, cfg['lr'], eps=cfg['epsilon'],
                                         weight_decay=cfg['weight_decay'])


    def _save_model(self, path):
        if not self.args['train']['model_saved_flag']:
            return
        self.model.save_non_clip_parameters(path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        info('1. Loading model from {}.'.format(path))
        self.model.load_non_clip_parameters(path)

    def get_proposal_results(self, pred_overlap, pred_reg, durations):
        # assume all valid scores are larger than one
        cfg = self.args
        cfg_anchor = cfg['model']['config']['simbase']['anchor']
        all_anchor_list = self.test_set.all_anchor_list.copy()
        overlap_result = [[] for _ in range(len(cfg_anchor['feature_map_len']))]
        reg_result = [[] for _ in range(len(cfg_anchor['feature_map_len']))]
        for i in range(len(cfg_anchor['feature_map_len'])):
            overlap_result[i].append(pred_overlap[i].detach().cpu().numpy())
            reg_result[i].append(pred_reg[i].detach().cpu().numpy())
        for i in range(len(cfg_anchor['feature_map_len'])):
            overlap_result[i] = np.concatenate(overlap_result[i], 0)
            reg_result[i] = np.concatenate(reg_result[i], 0)

        out_sorted_times = []
        for data_idx, duration in enumerate(durations):
            expand_anchor_list = []
            predict_overlap_list = []
            predict_center_list = []
            predict_width_list = []

            for anchor_group_id in range(len(cfg_anchor['feature_map_len'])):
                for anchor_id in range(cfg_anchor['feature_map_len'][anchor_group_id]):
                    for kk in range(self.args['model']['config']['simbase']['anchor_meta']['scale_ratios_num']):
                        expand_anchor_list.append(all_anchor_list[anchor_group_id][anchor_id][kk])
                        predict_overlap_list.append(overlap_result[anchor_group_id][data_idx][kk][anchor_id])
                        predict_center_list.append(reg_result[anchor_group_id][data_idx][kk*2][anchor_id])
                        predict_width_list.append(reg_result[anchor_group_id][data_idx][kk*2+1][anchor_id])


            a_left = []
            a_right = []
            a_score = []
            for index in range(len(predict_overlap_list)):
                anchor = expand_anchor_list[index]
                anchor_center = (anchor[1] - anchor[0]) * 0.5 + anchor[0]
                anchor_width = anchor[1] - anchor[0]

                if not self.args['model']['config']['simbase']['loss']['inference_regress_flag']:
                    p_center = anchor_center
                    p_width = anchor_width
                else:
                    center_offset = predict_center_list[index]
                    width_offset = predict_width_list[index]
                    p_center = anchor_center + self.args['model']['config']['simbase']['loss']['ratio_center'] * anchor_width * center_offset
                    p_width = anchor_width * np.exp(self.args['model']['config']['simbase']['loss']['ratio_width'] * width_offset)

                p_left = max(0, p_center - p_width * 0.5)
                p_right = min(cfg['dataset']['num_sample_clips'], p_center + p_width * 0.5)

                a_left.append(p_left)
                a_right.append(p_right)
                a_score.append(predict_overlap_list[index])

            picks = nms_temporal(a_left, a_right, a_score, 0.0)
            process_segment = []
            process_score = []
            for pick in picks:
                process_segment.append(
                    [float(a_left[pick]) / float(cfg['dataset']['num_sample_clips']) * durations[data_idx],
                     float(a_right[pick]) / float(cfg['dataset']['num_sample_clips']) * durations[data_idx],
                     ])
                process_score.append(a_score[pick])
            out_sorted_times.append(process_segment[:1])

        return out_sorted_times

def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
