FROM andrewosh/binder-base
MAINTAINER Alexander Panin <justheuristic@gmail.com>

RUN apt-get -qq update
RUN apt-get install -y cmake
RUN apt-get install -y zlib1g-dev
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl
RUN apt-get install -y libav-tools
RUN apt-get -y install swig #!This won't work with Box2D!

