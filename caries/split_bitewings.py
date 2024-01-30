from pathlib import Path

import json
import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import StratifiedGroupKFold


def determine_labels(
    coco: COCO,
) -> np.ndarray[np.int64]:
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
) -> np.ndarray[np.int64]:
    groups = []
    for img_dict in coco.imgs.values():
        patient = img_dict['file_name'].split('-')[1]
        groups.append(patient)

    _, groups = np.unique(groups, return_inverse=True)

    return groups


def split_bitewings(
    coco: COCO,
    n_folds: int=5,
) -> dict[str, np.ndarray[np.int64]]:
    labels = determine_labels(coco)
    groups = determine_groups(coco)

    splits = {}
    splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=1234)
    folds = [split[1] for split in list(splitter.split(coco.imgs, labels, groups))]

    for i in range(len(folds)):
        splits[f'train{i}'] = np.concatenate(
            folds[max(0, i - len(folds) + 1):max(0, i - 2)] +
            folds[i:i + len(folds) - 2]
        )
        splits[f'val{i}'] = folds[(i + len(folds) - 2) % len(folds)]
        splits[f'test{i}'] = folds[(i + len(folds) - 1) % len(folds)]

    return splits
    

def filter_coco(
    coco: COCO,
    idxs: np.ndarray[np.int64],
) -> dict[str, list]:
    coco_dict = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories'],
    }
    for i, img_dict in enumerate(coco.imgs.values()):
        if i not in idxs:
            continue

        coco_dict['images'].append(img_dict)

        anns = coco.imgToAnns[img_dict['id']]
        coco_dict['annotations'].extend(anns)

    return coco_dict
    
    
if __name__ == '__main__':
    root = Path(__file__).parent.parent.parent
    coco = COCO(root / 'annotations.json')
    splits = split_bitewings(coco, n_folds=10)

    out_dir = root / 'splits'
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idxs in splits.items():
        coco_dict = filter_coco(coco, idxs)
        with open(out_dir / f'{name}.json', 'w') as f:
            json.dump(coco_dict, f, indent=2)
