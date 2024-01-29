from pathlib import Path
from typing import List, Union

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CariesDataset(CocoDataset):

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'] + self.metainfo['attributes'],
        )

        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        class2label = {cat: i for i, cat in enumerate(self._metainfo['classes'])}
        attr2label = {cat: i for i, cat in enumerate(self._metainfo['attributes'])}
        for cat_id, cat in self.coco.cats.items():
            category = cat['name']
            if category in self._metainfo['classes']:
                class2label[cat_id] = class2label[category]

            if category in self._metainfo['attributes']:
                attr2label[cat_id] = attr2label[category]

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = Path(self.data_prefix['img']) / img_info['file_name']
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            instance['ignore_flag'] = ann.get('iscrowd', 0)
            instance['bbox'] = [x1, y1, x1 + w, y1 + h]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            if ann['category_id'] in class2label:
                instance['bbox_label'] = class2label[ann['category_id']]
            else:
                instance['bbox_label'] = class2label[ann['extra']['attributes'][0]]

            instance['bbox_multilabel'] = [0]*len(self._metainfo['attributes'])
            if ann['category_id'] in attr2label:
                label = attr2label[ann['category_id']]
                instance['bbox_multilabel'][label] = 1
            elif 'attributes' in ann['extra']:
                label = attr2label[ann['extra']['attributes'][0]]
                instance['bbox_multilabel'][label] = 1

            instances.append(instance)
        data_info['instances'] = instances
        
        return data_info
