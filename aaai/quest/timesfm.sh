salloc -A p32141 -p gengpu --mem=2G --gres=gpu:a100:1 -N 1 -n 1 -t 2:00:00 
srun --pty /bin/bash -l

ln -s /projects/[p...] ~/projects_shortcut

module load cuda/12.0.1-gcc-12.3.0
module load python-miniconda3
conda create --prefix projects_shortcut/pythonenvs/env1 python=3.10.12 --yes

conda env create --prefix projects_shortcut/pythonenvs/tfm_env --file=dynadojo/aaai/quest/tfm_env.yml

pip install timesfm

source activate projects_shortcut/pythonenvs/timesfm_env
conda install cudatoolkit
conda install cudnn
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


salloc -A p32141 -p gengpu --mem=2G --gres=gpu:a100:1 -N 1 -n 1 -t 2:00:00 
ssh [computer]
ln -s /projects/p32141 ~/projects_shortcut
module load python-miniconda3
# conda env create --prefix projects_shortcut/pythonenvs/tfm_env --file=dynadojo/aaai/quest/tfm_env.yml


#JAX ENV
conda create --prefix projects_shortcut/pythonenvs/tfm_env -c conda-forge -c nvidia python=3.10.12 jaxlib=*=*cuda* jax cuda-nvcc --yes