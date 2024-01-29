import copy
import pickle
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils
from scipy.interpolate import interp1d
import torch
from tqdm import tqdm      


def determine_fprs_sensitivities(
    results,
    match_labels: bool,
    match_attributes: bool,
    filter_labels: Optional[list[int]]=None,
    use_masks: bool=True,
    iou_thr: float=0.1,
):
    results = copy.deepcopy(results)
    if filter_labels is not None:
        filter_labels = torch.tensor(filter_labels)[:, None]
        for result in results:
            pred_keep = torch.any(
                result['pred_instances']['labels'] == filter_labels, axis=0,
            )
            gt_keep = torch.any(
                result['gt_instances']['labels'] == filter_labels, axis=0,
            )

            if 'masks' in result['pred_instances']:
                result['pred_instances']['masks'] = [
                    mask for mask, b in zip(result['pred_instances']['masks'], pred_keep) if b
                ]
            result['pred_instances']['labels'] = result['pred_instances']['labels'][pred_keep]
            result['pred_instances']['bboxes'] = result['pred_instances']['bboxes'][pred_keep]
            if 'multilogits' in result['pred_instances']:
                result['pred_instances']['multilogits'] = result['pred_instances']['multilogits'][pred_keep]
            result['pred_instances']['scores'] = result['pred_instances']['scores'][pred_keep]

            result['gt_instances']['masks'] = [mask for mask, b in zip(result['gt_instances']['masks'], gt_keep) if b]
            result['gt_instances']['labels'] = result['gt_instances']['labels'][gt_keep]
            result['gt_instances']['bboxes'] = result['gt_instances']['bboxes'][gt_keep]
            if 'multilabels' in result['gt_instances']:
                result['gt_instances']['multilabels'] = result['gt_instances']['multilabels'][gt_keep]
    

    total_preds = sum([result['pred_instances']['labels'].shape[0] for result in results])
    total_gts = sum([result['gt_instances']['labels'].shape[0] for result in results])
    offsets = [0, 0]

    ious = np.zeros((total_preds, total_gts))
    pred_scores = torch.zeros(total_preds)
    for result in results:
        if use_masks and 'masks' in result['pred_instances']:
            pred_masks = result['pred_instances']['masks']
            gt_masks = result['gt_instances']['masks']
        else:
            pred_masks = result['pred_instances']['bboxes'].numpy()
            gt_masks = result['gt_instances']['bboxes'].numpy()
            pred_masks[:, 2:] -= pred_masks[:, :2]
            gt_masks[:, 2:] -= gt_masks[:, :2]

        num_preds = len(pred_masks)
        num_gts = len(gt_masks)
        slices = (
            slice(offsets[0], offsets[0] + num_preds),
            slice(offsets[1], offsets[1] + num_gts),
        )
        result_ious = maskUtils.iou(pred_masks, gt_masks, [0]*len(gt_masks))
        result_ious = np.zeros((len(pred_masks), len(gt_masks))) if isinstance(result_ious, list) else result_ious
        ious[slices] = result_ious

        if match_labels:
            pred_labels = result['pred_instances']['labels']
            gt_labels = result['gt_instances']['labels']
            labels_mask = pred_labels[:, None] == gt_labels
            ious[slices] *= labels_mask.numpy()

        if match_attributes:
            pred_attributes = result['pred_instances']['multilogits'].argmax(-1)
            gt_attributes = result['gt_instances']['multilabels'].reshape(-1, 3).argmax(-1)
            attributes_mask = pred_attributes[:, None] == gt_attributes
            ious[slices] *= attributes_mask.numpy()

        pred_scores[slices[0]] = result['pred_instances']['scores']

        offsets[0] += num_preds
        offsets[1] += num_gts

    threshs = torch.cat((
        torch.tensor([0]), torch.sort(pred_scores)[0],
    )).numpy()

    fps, sensitivities = [], []
    for thresh in tqdm(threshs):
        pos_mask = pred_scores > thresh
        tp = np.any(ious[pos_mask] >= iou_thr, axis=0).sum()
        fp = pos_mask.sum() - tp
        sensitivity = tp / total_gts

        fps.append(fp)
        sensitivities.append(sensitivity)

    fprs = np.array(fps) / len(results)
    sensitivities = np.array(sensitivities)

    if fprs.max() < 8:
        fprs = np.concatenate((fprs, [8]))
        sensitivities = np.concatenate((sensitivities, [sensitivities[0]]))
        
    return fprs, sensitivities


def determine_auc(fprs, sensitivities, fpr_levels=[0.25, 0.5, 1, 2, 4, 8]):
    sensitivity_points = []
    for level in fpr_levels:
        idx = (fprs < level).argmax() - 1
        sensitivity_points.append(sensitivities[idx])

    return np.mean(sensitivity_points)


def froc(
    results,
    color,
    label,
    match_labels: bool=False,
    match_attributes: bool=False,
    filter_labels: Optional[list[int]]=None,
    use_masks: bool=True,
    fpr_levels=np.linspace(0, 8, 1001),
    ax=None,
):
    sensitivities_list, aucs = [], []
    for result in results:
        fprs, sensitivities = determine_fprs_sensitivities(
            result, match_labels, match_attributes, filter_labels, use_masks,
        )
        auc = determine_auc(fprs, sensitivities)
        aucs.append(auc)

        interp = interp1d(fprs, sensitivities)
        sensitivities = interp(fpr_levels)
        sensitivities_list.append(sensitivities)


    sensitivities = np.column_stack(sensitivities_list)
    aucs = np.array(aucs)

    means = sensitivities.mean(axis=-1)
    mins = np.maximum(0, sensitivities.mean(axis=-1) - sensitivities.std(axis=-1))
    maxs = np.minimum(1, sensitivities.mean(axis=-1) + sensitivities.std(axis=-1))
    std_text = '' if aucs.std() < 0.001 else f'±{aucs.std():.3f}'

    if ax is None:
        ax = plt.gca()
    ax.plot(fpr_levels, means, c=color, label=f'{label} (score={aucs.mean():.3f}{std_text})')
    ax.plot(fpr_levels, mins, c=color)
    ax.plot(fpr_levels, maxs, c=color)
    ax.fill_between(fpr_levels, mins, maxs, color=color, alpha=0.33)
    ax.grid()
    ax.set_xlabel('False positives per image')
    ax.legend()
    ax.set_ylabel('Sensitivity')


if __name__ == '__main__':
    template = 'work_dirs/fold{}_mask-rcnn_swin-t/detections.pkl'
    results = []
    for i in range(10):
        with open(template.format(i), 'rb') as f:
            result = pickle.load(f)
        
        results.append(result)

    # results = [results]
    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # froc(results, color=cycle[0], label='Caries')
    froc(results, color=cycle[1], label='Primary Caries', filter_labels=[0], use_masks=True, ax=axs[0])
    froc(results, color=cycle[2], label='Secondary Caries', filter_labels=[1], use_masks=True, ax=axs[1])
    # froc(results, color=cycle[1], label='Primary/Secondary Caries', match_labels=True)
    # froc(results, color=cycle[2], label='Caries Depth', match_attributes=True)
    plt.savefig('froc.png', dpi=500, bbox_inches='tight', pad_inches=0.0)
    plt.show()
