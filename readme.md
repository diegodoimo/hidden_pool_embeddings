# installation 
conda create --prefix ./env python=3.12 pip 
conda activate ./env

module load cuda/12.8 

pip install packaging wheel setuptools ninja


#this should be ok 
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

otherwise recompile with the following:
FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS=4 pip wheel flash-attn --no-build-isolation --no-deps -w ./wheels -v


pip install flash-attn==2.8.3 --no-build-isolation
%% pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

%% DS_BUILD_OPS=1 MAX_JOBS=8 pip install deepspeed==0.15.4 --no-build-isolation 

pip install -r requirements.txt