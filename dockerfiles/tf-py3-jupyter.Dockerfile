ARG DOCKER_ENV=cpu

FROM ulisesjeremias/tf-docker:${DOCKER_ENV}-jupyter
# DOCKER_ENV are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)

ARG DOCKER_ENV

ADD . /develop
COPY src /tf/notebooks

# Needed for string testing
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q
RUN apt-get install yasm
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y git nano
RUN apt-get install wget
RUN apt-get install tar

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python3 --version

RUN pip install --upgrade pip
RUN pip3 install sklearn opencv-python IPython
RUN if [[ "$DOCKER_ENV" = "gpu" ]]; then echo -e "\e[1;31mINSTALLING GPU SUPPORT\e[0;33m"; pip3 install -U tensorflow-gpu==2.0.0-beta1 tb-nightly; fi

WORKDIR /develop

# CMD ["bash", "-c", "source /etc/bash.bashrc"]
