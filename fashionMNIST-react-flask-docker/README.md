# Fashion MNIST Classifier - Front end: React, REST endpoints: Flask web app, Container: Docker, Deployment: AWS EC2

## Overview
As part of this sample project, I have created simple webapplication which will take input as image from the user and will do the inference at backend using pretrained deep learning model (fashionMNIST classifier). As a result of inference, it will send the category or class name of the image and finally output will be displayed to the user.

### Docker Compose
Docker maintains software and all of its dependencies within a "container", which can make collaborating and deploying simpler. 
Docker Compose is a tool for easily managing applications running multiple Docker containers.

Since the project involves development of two different applications, front end using React and  backend using Flask web app, we will need docker compose to containerise them and then finally deploying on AWS EC2 instance.

