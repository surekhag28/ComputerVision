# Linear regression from scratch using Ordinary Least Score - Front end: React, REST endpoints: Flask web app, Container: Docker, Deployment: AWS EC2

## Overview
As part of this sample project, I have created simple web application which will take input as csv file containing dataset from the user. This csv file will be further sent to flask web app which will perform linear regression on training data at backend side and will send output as root mean square error, r sqaured score and trained coefficients. These statistics are then finally displayed to user using react frontend app.

### Docker Compose
Docker maintains software and all of its dependencies within a "container", which can make collaborating and deploying simpler. 
Docker Compose is a tool for easily managing applications running multiple Docker containers.

Since the project involves development of two different applications, front end using React and  backend using Flask web app, we will need docker compose to containerise them and then finally deploying on AWS EC2 instance.

### Front end -- React App
At the front end side we are taking input as csv file from the user and by hitting the "Train" button, data will be sent to flask app where model will be trained using linear regression OLS. <br/>
In order to perform inference on unknown/test data, user is allowed to enter the "X" value and hit the "Predict" button.

### Back end -- Flask web app
Request from the user containing csv file as form data will be sent to concerned REST end point.
The service will upload csv file on the server and will train the model using linear regression OLS approach on train data.
The model will compute Root Mean Square error and R-Squared on test data for inference.
As a result/reponse it will send root mean square error, r sqaured score and trained coefficients to frontend app.
Also for inference, when user hits the Predict button, at the backend it will receive trained coefficients which will help in computing the predicted "Y" value.


### Deployment -- AWS EC2
Both the applications are dockerised using docker-compose and finally deployed on AWS EC2 instance. <br /><br />

***Note :- 1. Make sure to change the URLs with the Public DNS address in the application while sending request to the Flask web app. <br />
           2. Also ensure to implement the cross origin policy in the web app otherwise request from the react application will be blocked.***

#### Before sending the request to the server

![alt text](/LinearRegression_react_flask_docker_aws/images/pic1.png?raw=true)

#### After Training the model at backend side

![alt text](/LinearRegression_react_flask_docker_aws/images/pic2.png?raw=true)

#### After Predicting output for unknown/test data

![alt text](/LinearRegression_react_flask_docker_aws/images/pic3.png?raw=true)
