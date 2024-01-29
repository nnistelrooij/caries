from pathlib import Path
import tempfile

import json
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedGroupKFold


def determine_labels(
    coco: COCO,
):
    catid2label = {cat_id: i for i, cat_id in enumerate(coco.cats)}
    onehots = np.zeros((len(coco.imgs), len(coco.cats)), dtype=int)
    for i, img_id in enumerate(coco.imgs):
        for ann in coco.imgToAnns[img_id]:
            label = catid2label[ann['category_id']]
            onehots[i, label] = 1

    labels = 0
    for i, binary_label in enumerate(onehots.T):
        labels += 2 ** i * binary_label

    return labels


def determine_groups(
    coco: COCO,
):
    groups = []
    for img_dict in coco.imgs.values():
        patient = img_dict['file_name'].split('-')[1]
        groups.append(patient)

    _, groups = np.unique(groups, return_inverse=True)

    return groups


def split_prs(
    coco: COCO,
    n_folds: int=5,
    test: bool=False,
) -> dict:
    labels = determine_labels(coco)
    groups = determine_groups(coco)

    splits = {}
    splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=1234)
    folds = [split[1] for split in list(splitter.split(coco.imgs, labels, groups))]

    for i in range(len(folds)):
        splits[f'train{i}'] = np.concatenate(
            folds[max(0, i - len(folds) + 1):max(0, i - 1 - test)] +
            folds[i:i + len(folds) - 1 - test]
        )
        splits[f'val{i}'] = folds[(i + len(folds) - 1 - test) % len(folds)]
        if test:
            splits[f'test{i}'] = folds[(i + len(folds) - 1) % len(folds)]

    return splits


def swap_classes_attributes(
    coco,
):   
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': [
            {'name': cat, 'id': i + 1} for i, cat in enumerate('ABC')
        ],
    }
    for i, img_dict in enumerate(coco.imgs.values()):
        coco_dict['images'].append(img_dict)

        anns = coco.imgToAnns[img_dict['id']]
        for ann in anns:
            cat_name = coco.cats[ann['category_id']]['name']
            ann['category_id'] = 'ABC'.index(ann['extra']['attributes'][0]) + 1
            ann['extra']['attributes'][0] = cat_name
        
        coco_dict['annotations'].extend(anns)

    with tempfile.NamedTemporaryFile('w') as f:
        json.dump(coco_dict, f)
        f.flush()
        out = COCO(f.name)

    return out
    


def filter_coco(
    coco: COCO,
    idxs,
    filtered_coco: COCO,
    swap_classes_attributes: bool=False,
):
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': (
            coco.dataset['categories'] if not swap_classes_attributes else
            [{'name': cat, 'id': i + 1} for i, cat in enumerate('ABC')]
        ),
    }
    for i, img_dict in enumerate(coco.imgs.values()):
        if i not in idxs:
            continue

        if img_dict['id'] not in filtered_coco.imgs:
            continue

        coco_dict['images'].append(img_dict)

        anns = coco.imgToAnns[img_dict['id']]
        coco_dict['annotations'].extend(anns)

    return coco_dict
    
    
if __name__ == '__main__':
    root = Path('/home/mkaailab/.darwin/datasets/mucoaid/secondary-cariesv2')
    filter_anns = False
    swap = False

    coco_path = root / 'eduardo.json'
    coco = COCO(coco_path)
    if swap:
        coco = swap_classes_attributes(coco)
    splits = split_prs(coco, n_folds=10, test=True)

    if filter_anns:
        coco_path = root / 'aggregated.json'
        filtered_coco = COCO(coco_path)
    else:
        filtered_coco = coco

    for name, idxs in splits.items():
        coco_dict = filter_coco(coco, idxs, filtered_coco, swap)
        with open(root / f'{coco_path.stem}{"_swap" if swap else ""}_{name}.json', 'w') as f:
            json.dump(coco_dict, f, indent=2)
