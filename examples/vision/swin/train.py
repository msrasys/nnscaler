"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    examples/vision/swin/train.py
"""

import torch
from examples.vision.swin.model import SwinTransformer, ImageDataLoader

import cube
from cube.profiler.timer import CudaTimer, print_each_rank
from cube.profiler.memory import memory_summary, model_summary


class Config:

    # swin-large 201M
    embed_dim = 192
    depths = [2, 2, 18, 2]
    num_heads = [6, 12, 24, 48]

    # swin-huge: 2.5B
    # embed_dim = 512
    # depths = [2, 2, 42, 2]
    # num_heads = [16, 32, 64, 128]

    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None

    drop_path_rate = 0.2
    drop_rate = 0.2
    

    # 224 x 224
    img_size = 224
    window_size = 7

    # 640 x 640
    img_size = 640
    window_size = 40

    # 1536 x 1536
    # img_size = 1536
    # window_size = 48

    num_classes = 1000


def train():

    batch_size = 1

    cfg = Config()
    model = SwinTransformer(img_size=cfg.img_size,
                            patch_size=4,
                            in_chans=3,
                            num_classes=cfg.num_classes,
                            embed_dim=cfg.embed_dim,
                            depths=cfg.depths,
                            num_heads=cfg.num_heads,
                            window_size=cfg.window_size,
                            mlp_ratio=cfg.mlp_ratio,
                            qkv_bias=cfg.qkv_bias,
                            qk_scale=cfg.qk_scale,
                            drop_rate=cfg.drop_rate,
                            drop_path_rate=cfg.drop_path_rate,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False)

    model = model.cuda()
    dataloader = ImageDataLoader(batch_size, cfg.img_size, cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))

    print_each_rank('model weight consumpition:')
    memory_summary()

    def train_iter(model, dataloader):
        imgs, labels = next(dataloader)
        loss = model(imgs, labels)
        loss.backward()

    CudaTimer(enable=False).warmup()
    iter_num = 10
    for step in range(iter_num):

        if step == 0:
            model_summary(model, next(dataloader))

        if step >= 4:
            CudaTimer(enable=True).start('e2e')

        # training
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()

        if step >= 4:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('passed first iteration')
        
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-4, field_name='e2e')))
    memory_summary()

if __name__ == '__main__':

    cube.init()
    train()
