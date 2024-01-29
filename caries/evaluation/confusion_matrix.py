from collections import defaultdict
import copy
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from tqdm import tqdm


def determine_confusion_matrix(
    results,
    score_thrs: float=0.5,
    iou_thr: float=0.1,
    use_masks: bool=False,
):
    cm = np.zeros((3, 3), dtype=int)

    if isinstance(score_thrs, float) or score_thrs.ndim == 0:
        score_thrs = torch.tensor([score_thrs, score_thrs])

    pred_list, gt_list = {}, {}
    total_tp = 0
    for preds in results:
        if isinstance(preds, list):
            continue


        gt_instances = preds['gt_instances']
        gt_labels = gt_instances['labels']
        if gt_labels.numel() == 0:
            continue
        
        if use_masks and 'masks' in gt_instances:
            gt_bboxes = gt_instances['masks']
        else:
            gt_bboxes = gt_instances['bboxes'].numpy()
            gt_bboxes[:, 2:] -= gt_bboxes[:, :2]


        pred_instances = preds['pred_instances']
        keep = pred_instances['scores'] >= score_thrs[pred_instances['labels']]
        pred_labels = pred_instances['labels'][keep]
        pred_instances['bboxes'] = pred_instances['bboxes'][keep]
        if pred_labels.numel() == 0:
            for label in gt_labels:
                cm[label, -1] += 1

            continue
        
        if use_masks and 'masks' in pred_instances:
            pred_bboxes = pred_instances['masks']
            pred_bboxes = [bbox for bbox, b in zip(pred_bboxes, keep) if b]
        else:
            pred_bboxes = pred_instances['bboxes'].numpy()
            pred_bboxes[:, 2:] -= pred_bboxes[:, :2]


            
        # compute iou between each dt and gt region
        iscrowd = [0 for o in gt_bboxes]
        ious = maskUtils.iou(pred_bboxes, gt_bboxes, iscrowd)  # pred x gt


        tp = ious > iou_thr
        total_tp += tp.sum()
        fp = np.all(ious <= iou_thr, axis=1)
        fn = np.all(ious <= iou_thr, axis=0)

        for idxs in np.column_stack(np.nonzero(tp)):
            cm[gt_labels[idxs[1]], pred_labels[idxs[0]]] += 1

        for idx in np.nonzero(fp)[0]:
            cm[-1, pred_labels[idx]] += 1

        for idx in np.nonzero(fn)[0]:
            cm[gt_labels[idx], -1] += 1
            
    return cm


def compute_metrics(cm, labels):
    metrics = defaultdict(list)
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        tn = cm.sum() - tp - fp - fn

        metrics[f'{label}_precision/ppv'] = tp / (tp + fp)
        metrics['precision/ppv'] = metrics['precision/ppv'] + [tp / (tp + fp)]
        metrics[f'{label}_sensitivity/recall'] = tp / (tp + fn)
        metrics['sensitivity/recall'] = metrics['sensitivity/recall'] + [tp / (tp + fn)]
        metrics[f'{label}_specificity'] = tn / (tn + fp)
        metrics['specificity'] = metrics['specificity'] + [tn / (tn + fp)]
        metrics[f'{label}_npv'] = tn / (tn + fn)
        metrics['npv'] = metrics['npv'] + [tn / (tn + fn)]
        metrics[f'{label}_f1'] = 2 * tp / (2 * tp + fp + fn)
        metrics['f1'] = metrics['f1'] + [2 * tp / (2 * tp + fp + fn)]

    for key, value in metrics.items():
        if not isinstance(value, list):
            continue

        metrics[key] = np.mean(value)

    return metrics

    


def f1_scores(cm):
    f1s = []
    for label in range(2):
        precision = cm[label, label] / cm[label].sum()
        recall = cm[label, label] / cm[:, label].sum()
        f1 = (2 * precision * recall) / (precision + recall)
        f1s.append(f1)

    return np.array(f1s)


def determine_optimal_cm(
    results,
    use_masks: bool,
):
    threshs = torch.linspace(0, 1, 101)
    f1s = np.zeros((0, 2))
    for thresh in tqdm(threshs):
        results2 = copy.deepcopy(results)
        cm = determine_confusion_matrix(results2, thresh, use_masks=use_masks)
        f1s = np.concatenate((
            f1s, [f1_scores(cm)],
        ))

    threshs = threshs[np.nanargmax(f1s, axis=0)]
    print(threshs)

    return determine_confusion_matrix(results, threshs, use_masks=use_masks)


def draw_confusion_matrix(
    cm,
    labels,
    ax,
    yaxis: bool=True,
    colorbar: bool=True,
    recolor: bool=False,
):
    if not recolor:
        norm_cm = cm / cm.sum(axis=0, keepdims=True)
    else:
        norm_cm = cm

    disp = ConfusionMatrixDisplay(norm_cm, display_labels=labels)
    disp.plot(cmap='magma', ax=ax, colorbar=colorbar)

    # draw colorbar according to largest non-TN value
    if recolor:
        cm_without_tn = cm.copy()
        cm_without_tn[-1, -1] = 0
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=cm_without_tn.max())
        disp.ax_.images[0].set_norm(normalize)
        disp.text_[0, 0].set_color(disp.im_.cmap(0.0))
    else:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                disp.text_[i, j].set_text(cm[i, j])
        
        normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)
        disp.ax_.images[0].set_norm(normalize)

    if not yaxis:
        disp.ax_.yaxis.set_visible(False)
    
    # draw y ticklabels vertically
    offset = matplotlib.transforms.ScaledTranslation(-0.1, 0, disp.figure_.dpi_scale_trans)
    for label in disp.ax_.get_yticklabels():
        label.set_rotation(90)
        label.set_transform(label.get_transform() + offset)
        label.set_ha('center')
        label.set_rotation_mode('anchor')


labels = ['Primary Caries', 'Secondary Caries', 'No caries']
metrics = {}
cm = np.zeros((3, 3), dtype=int)
for fold in range(10):

    with open(f'work_dirs/fold{fold}_mask-rcnn_swin-t/detections.pkl', 'rb') as f:
        results = pickle.load(f)

    single_cm = determine_optimal_cm(results, use_masks=True)
    cm += single_cm
    for key, value in compute_metrics(single_cm, labels[:-1]).items():
        if key in metrics:
            metrics[key].append(value)
            continue

        metrics[key] = [value]

print('macro-average+-std')
for key, values in metrics.items():
    m, s = np.mean(values), np.std(values)
    print(f'{key}:{m:.5f}+-{s:.5f}')

print('\nmicro-average')
metrics = compute_metrics(cm, labels[:-1])
for key, value in metrics.items():
    print(f'{key}:{value:.5f}')

fig, ax = plt.subplots(1, 1)
draw_confusion_matrix(cm, labels=labels, ax=ax, recolor=True)

plt.savefig(f'cm.png', dpi=500, bbox_inches='tight', pad_inches=0.0)
plt.show()
