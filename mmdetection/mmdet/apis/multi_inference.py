import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
import os
import sys

import cv2
from tqdm import tqdm
import argparse
from mmcv.parallel import scatter, collate, MMDataParallel
from mmdet.core import get_classes
from mmdet.datasets import to_tensor
from mmdet.datasets.transforms import ImageTransform
from mmdet.models import build_detector, detectors
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict


def _prepare_data(img, img_transform, cfg, flip, trans_scale, device):
    if flip: img = cv2.flip(img, 1)
    ori_shape = img.shape
    height, width = img.shape[:2]
    transform_scale = trans_scale
    img, img_shape, pad_shape, scale_factor = img_transform(
        img,
        scale=transform_scale,
        keep_ratio=cfg.data.test.get('resize_keep_ratio', True))
    img = to_tensor(img).to(device).unsqueeze(0)
    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=False)
    ]
    # print(img_meta)
    return dict(img=[img], img_meta=[img_meta])


def flip_im_detect(im, det_f):
    det_t = []
    for det_cls in det_f:
        if det_cls.shape[0] == 0:
            det_t.append(det_cls)
        else:
            det_cls_t = np.zeros(det_cls.shape)
            # print(det_cls_t.shape)
            det_cls_t[:, 0] = im.shape[1] - det_cls[:, 2]
            det_cls_t[:, 1] = det_cls[:, 1]
            det_cls_t[:, 2] = im.shape[1] - det_cls[:, 0]
            det_cls_t[:, 3] = det_cls[:, 3]
            det_cls_t[:, 4] = det_cls[:, 4]
            det_t.append(det_cls_t)
    return det_t


def _inference_single(model, img, img_transform, cfg, flip, scale, device):
    img = mmcv.imread(img)
    data = _prepare_data(img, img_transform, cfg, flip, scale, device)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        if flip: result = flip_im_detect(img, result)
    return result


def _inference_generator(model, imgs, img_transform, cfg, flip, scale, device):
    for img in imgs:
        yield _inference_single(model, img, img_transform, cfg, flip, scale, device)


def inference_detector(model, imgs, cfg, flip, scale, device='cuda:0'):
    img_transform = ImageTransform(
        size_divisor = cfg.data.test.size_divisor, **cfg.img_norm_cfg)
    model = model.to(device)
    model.eval()

    if not isinstance(imgs, list):
        return _inference_single(model, imgs, img_transform, cfg, flip, scale, device)
    else:
        return _inference_generator(model, imgs, img_transform, cfg, flip, scale, device)


def show_result(img, result, dataset='coco', score_thr=0.3, out_file=None):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    # draw segmentation masks
    if segm_result is not None:
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        for i in inds:
            color_mask = np.random.randint(
                0, 256, (1, 3), dtype=np.uint8)
            mask = maskUtils.decode(segms[i]).astype(np.bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr,
        show=out_file is None)



def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('--config_file', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--image_dir', help='inference image dir')
    parser.add_argument('--output_dir', help='output results dir')
    parser.add_argument('--multi',default=True,type=bool,help='if or not multi inference')
    parser.add_argument('--multi_list',help="multi list ratio for inference")
    parser.add_argument('--flip',default=False,type=bool,help='if or not flip')
    parser.add_argument('--split_txt', help="image order txt")
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    # parser.add_argument('--out_name', help='output result file')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # if args.out_name is not None and not args.out_name.endswith(('.pkl', '.pickle')):
    #     raise ValueError('The output file must be a pkl file.')
    cfg = mmcv.Config.fromfile(args.config_file)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    split_txt = args.split_txt
    image_name_list = []
    file_open = open(split_txt, 'r')
    lines = file_open.readlines()
    for line in lines:
        name = line.strip()
        image_name_list.append(name)

    file_open.close()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])
        if args.multi:
            origin_image_dir = args.image_dir
            # scale_list = [(1333, 800), (1000, 600), (667, 400), (333, 200)]
            # scale_list = [(1600, 800), (1600, 1000), (1600, 600), (1600, 1200)] 
            # scale_list = [(1600, 1000), (1600, 800)]
            scale_list = [(1600, 800)]

            for scale in scale_list:
                print(scale)
                scale_dir = origin_image_dir
                scale_dir += "/"
                result_list = []

                for i in tqdm(range(len(image_name_list))):
                    name = str(image_name_list[i])+".jpg"
                    image_path = os.path.join(scale_dir, name)

                    result = inference_detector(model, image_path, cfg, False, scale, device='cuda:0')

                    if args.flip:
                        result_f = inference_detector(model, image_path, cfg, True, scale, device='cuda:0')
                        if len(result_f) != len(result):
                            raise ValueError('Inference classes must be equal!')
                        for k in range(len(result)):
                            result[k] = np.row_stack([result[k], result_f[k]])

                    result_list.append(result)

                output_path = args.output_dir + '/result' + str(scale[1]) + '.pkl'
                mmcv.dump(result_list, output_path)


if __name__=="__main__":
    print("start multi inference and get the final results")
    main()
