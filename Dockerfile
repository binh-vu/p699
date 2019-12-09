FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN pip install librosa
RUN apt install -y libsndfile-dev ffmpeg