# python scripts/train_conditional_seismic_imaging.py --dataset seismic --modes 24 --cuda 0 --hidden_dim 64  --batchsize 32 --val_batchsize 32 --max_epochs 1000 &
# python scripts/train_conditional_seismic_imaging.py --dataset seismic --modes 12 --cuda 1 --hidden_dim 128 --batchsize 32 --val_batchsize 32 --max_epochs 1000 &
python scripts/train_conditional_seismic_imaging.py --dataset seismic --modes 24  --cuda 2 --hidden_dim 32  --batchsize 128 --val_batchsize 128 --max_epochs 2000 &

