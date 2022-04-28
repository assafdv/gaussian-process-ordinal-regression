FROM continuumio/miniconda3:4.6.14

# install dependencies (this is the base docker)
RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install python3-dev -y
RUN apt-get install build-essential -y  
RUN conda install cython=0.29.13 numpy=1.18.5 scipy=1.3.1 matplotlib=3.1.0 scikit-learn=0.21.3
RUN conda install pip
RUN pip install gpy==1.9.8
RUN pip install ddt==1.4.4
# Add the application source code.
ADD . /app
WORKDIR /app

