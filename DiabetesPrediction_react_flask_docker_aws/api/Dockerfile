FROM anibali/pytorch:cuda-10.0
USER root
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN mkdir /backend
WORKDIR /backend
COPY requirements.txt /backend/requirements.txt
RUN pip install --upgrade pip && \
	pip install -r requirements.txt
COPY . .
EXPOSE 5000

