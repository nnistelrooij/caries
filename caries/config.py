_base_ = '../mmdetection/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'

custom_imports = dict(
    imports=['caries.evaluation.metrics', 'caries.det_tta'],
    allow_failed_imports=False,
)

data_root = './'
data_prefix = dict(img=data_root + 'images')

split = 0
work_dir = f'work_dirs/split{split}/'

classes = ['Primary Caries', 'Secondary Caries']
filter_empty = False

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
                'type':
                'RandomChoiceResize',
                'scales': [(480, 1333), (512, 1333), (544, 1333),
                            (576, 1333), (608, 1333), (640, 1333),
                            (672, 1333), (704, 1333), (736, 1333),
                            (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
            [{
                'type': 'RandomChoiceResize',
                'scales': [(400, 1333), (500, 1333),
                            (600, 1333)],
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_type': 'absolute_range',
                'crop_size': (384, 600),
                'allow_negative_crop': True
            }, {
                'type':
                'RandomChoiceResize',
                'scales':
                [(480, 1333), (512, 1333), (544, 1333),
                    (576, 1333), (608, 1333), (640, 1333),
                    (672, 1333), (704, 1333), (736, 1333),
                    (768, 1333), (800, 1333)],
                'keep_ratio':
                True
            }],
        ]),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='CocoDataset',
    filter_cfg=dict(filter_empty_gt=filter_empty),
    serialize_data=False,
    pipeline=train_pipeline,
    ann_file=f'splits/train{split}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes),
))

val_pipeline=[
    dict(
        type='LoadImageFromFile',
    ),
    dict(
        type='Resize',
        scale=(1333, 800),
        keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

val_dataloader = dict(dataset=dict(
    ann_file=f'splits/val{split}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes),
    pipeline=val_pipeline,
))
val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=f'splits/val{split}.json',
    metric=['bbox', 'segm'],
)

test_dataloader = dict(dataset=dict(
    ann_file=f'splits/test{split}.json',
    data_prefix=data_prefix,
    data_root=data_root,
    metainfo=dict(classes=classes),
    pipeline=val_pipeline,
))
test_evaluator = dict(
    _delete_=True,
    type='DumpGTPredDetResults',
    out_file_path=work_dir + 'detections.pkl',
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)),
    ),
    test_cfg=dict(rcnn=dict(score_thr=0.0)),
)
load_from = 'checkpoints/mask-rcnn_swin-t.pth'

max_epochs = 36
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    )
]

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=0.00005,
    betas=(0.9, 0.999),
    weight_decay=0.1,
))
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        by_epoch=True,
        max_keep_ckpts=1,
        save_best='coco/segm_mAP',
        rule='greater',
    ),
    visualization=dict(
        draw=True,
        interval=20,
    ),
)

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TestTimeAug', transforms=[
        [
            {
                'type': 'Resize',
                'scale': scale,
                'keep_ratio': True,
            } for scale in [
                (1333, 640), (1333, 672), (1333, 704),
                (1333, 736), (1333, 768), (1333, 800),
                # (1333, 800),
            ]
        ],
        [
            {'type': 'RandomFlip', 'prob': 0.0},
            {'type': 'RandomFlip', 'prob': 1.0},
        ],
        [{
            'type': 'LoadAnnotations', 'with_bbox': True, 'with_mask': True,
        }],
        [{
            'type': 'PackDetInputs',
            'meta_keys': [
                'img_id', 'img_path', 'ori_shape', 'img_shape',
                'scale_factor', 'flip', 'flip_direction',
            ]
        }],
    ]),
]

tta_model = dict(
    type='InstSegTTAModel',
    tta_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
    ),
)
