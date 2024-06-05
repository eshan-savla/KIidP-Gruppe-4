FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
# Set timezone info
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Install dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive && apt-get install git ffmpeg libsm6 libxext6  -y && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/
RUN echo "numpy \nopencv-python \nmatplotlib \nscikit-image \nimageio \ntorchvision \ntorchsummary \ntensorboardX \npyrealsense2 \nPillow" > requirements.txt
RUN pip install -r requirements.txt 

CMD [ "bash" ]