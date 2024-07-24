#!/bin/bash

#SBATCH -p gpu20
#SBATCH -t 0-12:00:00
#SBATCH -o /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.out
#SBATCH -e /CT/HOIMOCAP/work/exps/slurm_outputs/%A_%a.err
#SBATCH -a 1
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 18000

# setup the slurm
#. ./slurmSetup.sh

echo $PWD

eval "$(conda shell.bash hook)"
echo "Activate metacap environment"
conda activate metacap


echo python launch.py --config configs/s5/metalearning/metaneusseq-domedenserawseq-ddc-s5-smooth2-24-more-newthreshold0-pointcloud-zero4.yaml --gpu 0 --train tag=Test_Down2.0_Blur_100Frame_100Camera_DDC dataset.img_downscale=2.0 dataset.blur=True dataset.preload=True dataset.blur=True dataset.with_depth=True model.decay_step=300 dataset.depth_shift=-0.1 system.loss.lambda_eikonal=0.5 dataset.smoothDQ=True
python launch.py --config configs/s5/metalearning/metaneusseq-domedenserawseq-ddc-s5-smooth2-24-more-newthreshold0-pointcloud-zero4.yaml --gpu 0 --train tag=Test_Down2.0_Blur_100Frame_100Camera_DDC dataset.img_downscale=2.0 dataset.blur=True dataset.preload=True dataset.blur=True dataset.with_depth=True model.decay_step=300 dataset.depth_shift=-0.1 system.loss.lambda_eikonal=0.5 dataset.smoothDQ=True
