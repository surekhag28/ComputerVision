FROM anibali/pytorch:cuda-10.0
COPY . /usr/app/
EXPOSE 8000
WORKDIR /usr/app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD python app.py
