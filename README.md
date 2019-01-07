# gan_inpainting_baseline

Definition for runtime options

- Pretrain baseline policy for e.g. 10000 iterations using CelebA dataset
```commandline
python baseline.py -f pretrain -i 10000 -d celeba
```
- Train the entire GAN model for e.g. 20000 iterations with batch size 64 using ImageNet dataset
```commandline
python baseline.py -f train -i 20000 -b 64 -d imagenet
```
- Train the entire GAN model without using the previous saved checkpoints (**this will reset the checkpoint directory by the end of the training**)
```commandline
python baseline.py -p False
```