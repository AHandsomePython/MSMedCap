python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_opt2.7b_eval.yaml

python evaluate.py --cfg-path lavis/projects/blip2/eval/caption_coco_opt2.7b_eval.yaml