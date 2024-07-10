FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev git-lfs ffmpeg libsm6 libxext6 cmake \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* && git lfs install

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
	PYTHONUNBUFFERED=1 \
	GRADIO_ALLOW_FLAGGING=never \
	GRADIO_NUM_PORTS=1 \
	GRADIO_SERVER_NAME=0.0.0.0 \
	GRADIO_THEME=huggingface \
    GRADIO_SHARE=False \
	SYSTEM=spaces

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install specific versions of PyTorch and TorchVision
RUN pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
RUN pip install --no-cache-dir gradio==3.50.2 opencv-python supervision==0.17.0rc4 \
    pillow requests setofmark==0.1.0rc3

# Install SAM and Detectron2
RUN pip install 'git+https://github.com/facebookresearch/segment-anything.git'
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Download weights
RUN mkdir -p $HOME/app/weights
RUN wget -c -O $HOME/app/weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY app.py .
COPY utils.py .
COPY sam_utils.py .

RUN find $HOME/app

# Set the environment variable to specify the GPU device
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# Run your app.py script
CMD ["python", "app.py"]
