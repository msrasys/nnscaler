Prepare data:
```
python nanoGPT/data/shakespeare_char/prepare.py
```

Run without nnscaler
```
python train_lightning.py nanoGPT/config/train_shakespeare_char.py
```

Run with nnscaler
```
torchrun --standalone --nproc_per_node=1  train_lightning.py nanoGPT/config/train_shakespeare_char.py
```
