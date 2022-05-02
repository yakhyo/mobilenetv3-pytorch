torchrun --nproc_per_node=2 train.py --epochs 300  --batch-size 128 --lr 0.064 --weight-decay 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2 --resume weights/last.pth


#torchrun --nproc_per_node=2 train.py --epochs 300  --batch-size 128 --lr 0.064 --weight-decay 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2 --resume weights/last.pth