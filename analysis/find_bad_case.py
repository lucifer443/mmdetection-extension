from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import inference_detector
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from pathlib import Path

import os
import mmcv
import math
import torch
import pickle as pkl


def init_detector(config, checkpoint=None, device='cuda:0'):
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg, train_cfg=config.train_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def draw_bbox_with_gt(detector, img, gt_bboxes, gt_labels, score_thr=0.3, show=True):
    output_name = os.path.join("tmp/",
                            Path(img).name)
    preds = inference_detector(detector, img)
    img = detector.show_result(img, preds, score_thr=score_thr, show=False)
    mmcv.imshow_det_bboxes(img,
                           gt_bboxes,
                           gt_labels,
                           class_names=["p", "h", "c", "l"],
                           show=show,
                           out_file=output_name,
                           bbox_color='red',
                           text_color='red',)


def evaluate_case(detector, data_loader):
    loss_dict = dict()
    nan_list = []
    device = next(detector.parameters()).device
    prog_bar = mmcv.ProgressBar(len(data_loader.dataset))
    for i, data in enumerate(data_loader):
        if isinstance(data["img"], list):
            for key in data.keys():
                data[key] = data[key][0]
        filename = data['img_metas'].data[0][0]["filename"]
        data = scatter(data, [device])[0]
        with torch.no_grad():
            loss = detector(return_loss=True, **data)
        # if i > 10:
        #     break
        total_loss = sum([sum(loss[key]).cpu().numpy() for key in loss])
        if math.isnan(total_loss) or math.isinf(total_loss):
            nan_list.append(filename)
        else:
            loss_dict[filename] = total_loss
        prog_bar.update()

    print(f"\nmax loss is {max(loss_dict.values())}, average loss is {sum(loss_dict.values())/len(loss_dict)}")
    print(f"numbers of nan loss: {len(nan_list)}")
    with open("analysis_with_loss.pkl", "wb") as f:
        pkl.dump({"loss": loss_dict, "nan_list": nan_list}, f)


if __name__ == "__main__":
    cfg = Config.fromfile('configs/cm_1.py')
    detector = init_detector(cfg.deepcopy(), checkpoint="pretrained/cm_1-68caab03.pth", device="cuda:0")
    cfg.data.test.pipeline.insert(1, dict(type='LoadAnnotations', with_bbox=True))
    cfg.data.test.pipeline[-1]["transforms"][-1]["keys"].extend(['gt_bboxes', 'gt_labels'])

    ds = build_dataset(cfg.data.test)
    dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    # evaluate_case(detector, dl)

    with open("analysis_with_loss.pkl", "rb") as f:
        data = pkl.load(f)

    # import matplotlib.pyplot as plt
    #
    # loss直方图
    # plt.hist(data["loss"].values(), bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show()
    import numpy as np
    mean, var = np.mean(list(data["loss"].values())), np.var(list(data["loss"].values()))
    print(mean, var, mean + 10*var)
    visualize_list = data["nan_list"].copy()
    for key, value in data['loss'].items():
        if value > mean + 10*var:
            visualize_list.append(key)

    print(len(visualize_list))
    raw_data_cfg = cfg.data.test.deepcopy()
    raw_data_cfg.pipeline = raw_data_cfg.pipeline[:2] + \
                            [dict(type='Collect',
                                  keys=['img', 'gt_bboxes', 'gt_labels'],
                                  meta_keys=["filename"])]
    print(raw_data_cfg.pipeline)
    for item in build_dataset(raw_data_cfg):
        filename = item['img_metas'].data['filename']
        if filename in visualize_list:
            print(filename, data["loss"].get(filename, float("nan")))
            draw_bbox_with_gt(detector, filename, item['gt_bboxes'], item['gt_labels'], show=False)

