FROM tensorflow/tensorflow:1.15.0-gpu-py3-jupyter

RUN apt update -y && apt install -y libsndfile-dev ffmpeg

RUN apt install -y nodejs npm git
RUN pip install librosa tqdm matplotlib jupyter jupyterlab
RUN pip install ipywidgets \
    && jupyter nbextension enable --py widgetsnbextension \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager

RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
    && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
    && apt update -y && apt install -y yarn

RUN cd /tmp && git clone https://github.com/minhptx/jupyterlab-monaco/ \
    && cd jupyterlab-monaco && yarn install && yarn run build && jupyter labextension link .

RUN pip install ujson
ADD jupyter_notebook_config.json /root/.jupyter/