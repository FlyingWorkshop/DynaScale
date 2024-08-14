# TO INSTALL
#request a gpu node
sh_dev -g 1 
module load cuda/12.1.1
module load python/3.9.0
mkdir -p $GROUP_HOME/dynadojo/mamba
PYTHONUSERBASE=$GROUP_HOME/dynadojo/mamba pip install --user git+git@github.com:state-spaces/mamba.git@62db608


# TO RUN NOTEBOOK
module load cuda/12.1.1
module load python/3.9.0
module load py-ipython/8.3.0_py39
export PYTHONPATH=$GROUP_HOME/dynadojo/mamba/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH
