## Implementation of [MobileNetV3](https://arxiv.org/abs/1905.02244) in PyTorch

**Arxiv**: https://arxiv.org/abs/1905.02244

After 300 epochs MobileNetV3L reaches **Acc@1**: 74.3025 **Acc@5**: 91.8342

### Updates

* 2022.05.13:
    - Weights are uploaded to the `weights` folder. `last.ckpt` is checkpoint (88.3MB) (includes model, model_ema, optimizer, ...) and last.pth is model with
      Exponential Moving Average (11.2MB) and converted to `half()` tensor.

### Dataset

Specify the IMAGENET data folder in the `main.py` file.

``` python
parser.add_argument("--data-path", default="../../Projects/Datasets/IMAGENET/", type=str, help="dataset path")
```

IMAGENET folder structure:

```
├── IMAGENET 
    ├── train
         ├── [class_id1]/xxx.{jpg,png,jpeg}
         ├── [class_id2]/xxy.{jpg,png,jpeg}
         ├── [class_id3]/xxz.{jpg,png,jpeg}
          ....
    ├── val
         ├── [class_id1]/xxx1.{jpg,png,jpeg}
         ├── [class_id2]/xxy2.{jpg,png,jpeg}
         ├── [class_id3]/xxz3.{jpg,png,jpeg}
```

#### Augmentation:

`AutoAugment` for IMAGENET is used as a default augmentation. The interpolation mode is `BILINEAR`

### Train

Run `main.sh` (for DDP) file by running the following command:

```
bash main.sh
```

`main.sh`:

```
torchrun --nproc_per_node=@num_gpu main.py --epochs 300  --batch-size 512 --lr 0.064  --lr-step-size 2 --lr-gamma 0.973 --random-erase 0.2
```

Run `main.py` for `DataParallel` training.

The training config taken
from [official torchvision models' training config](https://github.com/pytorch/vision/tree/970ba3555794d163daca0ab95240d21e3035c304/references/classification)
.

### Evaluation
```commandline
python main.py --test
```
