module purge
module load gnu/8.4.0 cuda/10.2.89
#export MV2 PATH
# I will recommmend you to install a new RPM and update the mpicc and mpicxx files
# there is not need to update "/user/local/cuda/10.2" thing
source miniconda3/bin/activate
conda create -n your-name python=3.6.5
conda activate your-name
export PYTHONNOUSERSITE=true
pip install torch torchvision
HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_CUDA_HOME=/usr/local/cuda/10.2.89 HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod
#Running
#hosts file has hostnames
#add this to your .bashrc (just for this project, you should remove it later)
export LD_LIBRARY_PATH=/usr/local/cuda/10.2.89/lib64:$LD_LIBRARY_PATH
mpiexec -n 2 -hostfile hosts -genv MV2_USE_CUDA=1 -genv MV2_SUPPORT_DL=1 python wikiData.py
export LD_PRELOAD=/replace-with-your-path-to-Tau/Tau/tau-2.30/x86_64/lib/shared-ompt-v5-mpi-pdt-openmp/libTAU.so
export MV2_PATH=/replace-with-your-path-to-MPI/mv2/opt/mvapich2/gdr/2.3.4/mcast/no-openacc/cuda10.2/mofed4.7/mpirun/gnu8.4.0/bin
mpirun -np 1 tau_exec -T mpi,openmp,ompt,v5,pdt -ompt python replace-with-your-training-script
