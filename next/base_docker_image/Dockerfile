FROM python:2
MAINTAINER Lalit Jain, lalitkumarj@gmail.com

# Install python dependencies for next_backend
RUN pip install -U setuptools
RUN pip install ipdb==0.10.2
RUN pip install line_profiler==2.0
RUN pip install celery==3.1.25

# Install python dependencies for next_backend
ADD requirements.txt /requirements.txt
RUN pip install -U -r requirements.txt
