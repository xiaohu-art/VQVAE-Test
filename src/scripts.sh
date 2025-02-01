# Train a VQ-VAE on ImageNet
# CUDA_VISIBLE_DEVICES=0 python src/train.py  \
#    --epoch 100 \
#    --lr 1e-4 \
#    --weight_decay 1e-4 \
#    --opt adam \
#    --batch_size 4 \
#    --dropout 0.0 \
#    --seed 0 \
#    --save_dir vqvae \
#    --warmup_iters 10 \
#    --decay_iters 50 \
#    --dataset imagenet \
#    --model vqvae \
#    --codebook cosine     # euclidean

# Train a VQ-VAE with the rotation trick..
CUDA_VISIBLE_DEVICES=0 python src/train.py \
 --epoch 100 \
 --lr 1e-4 \
 --weight_decay 1e-4 \
 --opt adam \
 --batch_size 4 \
 --dropout 0.0 \
  --seed 0 \
  --save_dir rot_vqvae \
  --warmup_iters 10 \
  --decay_iters 50 \
  --dataset imagenet \
  --model rot_vqvae \
  --codebook cosine
