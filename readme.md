# installation 
conda create --prefix ./env python=3.12 pip 
conda activate ./env

module load cuda/12.8 
export CUDA_HOME=$CUDA_PATH


pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install vllm= --extra-index-url https://download.pytorch.org/whl/cu126
DS_BUILD_OPS=1 python -m build --wheel --no-isolation --config-setting="--build-option=build_ext" --config-setting="--build-option=-j8"

pip install packaging
pip install -r requirements.txt