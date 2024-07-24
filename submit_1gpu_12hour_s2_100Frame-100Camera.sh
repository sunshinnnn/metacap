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

echo python launch.py --config configs/s2/metalearning/metaneusseq-domedenserawseq_meta-ddc-s2-smooth2-24-more-newthreshold0-100Frame-100Camera.yaml --gpu 0 --train tag=Test_Down2.0_Blur_100Frame_100Camera_DDC dataset.img_downscale=2.0 dataset.blur=True dataset.preload=True
python launch.py --config configs/s2/metalearning/metaneusseq-domedenserawseq_meta-ddc-s2-smooth2-24-more-newthreshold0-100Frame-100Camera.yaml --gpu 0 --train tag=Test_Down2.0_Blur_100Frame_100Camera_DDC dataset.img_downscale=2.0 dataset.blur=True dataset.preload=True

