# Diabetes Predictor using RandomForest Regression - Front end: React, REST endpoints: Flask web app, Container: Docker, Deployment: AWS EC2

## Overview
As part of this project, I have created Diabetes predictor using RandomForest regression which will take certain values from the user via frontend application and will send the inference output back to the them.

### Docker Compose
Docker maintains software and all of its dependencies within a "container", which can make collaborating and deploying simpler. 
Docker Compose is a tool for easily managing applications running multiple Docker containers.

Since the project involves development of two different applications, front end using React and  backend using Flask web app, we will need docker compose to containerise them and then finally deploying on AWS EC2 instance.

### Front end -- React App
At the front end side we are taking input values from the user and send it to the backend server via REST endpoints.
After performing inference, it sends response back to the React application which finally displays predicted output as either :"Diabetes" or "No Diabetes" 
depending on the given input.

### Back end -- Flask web app
After receiving the request and input values from the user, it loads pretrained machine learning model to perform inference on the test data and sends response 
back in json format to the React App.


### Deployment -- AWS EC2
Both the applications are dockerised using docker-compose and finally deployed on AWS EC2 instance. <br /><br />

***Note :- 1. Make sure to change the URLs with the Public DNS address in the application while sending request to the Flask web app. <br />
           2. Also ensure to implement the cross origin policy in the web app otherwise request from the react application will be blocked.***

#### Before sending the request to the server

![alt text](/DiabetesPrediction_react_flask_docker_aws/images/pic1.png?raw=true)

#### Input the values

![alt text](/DiabetesPrediction_react_flask_docker_aws/images/pic2.png?raw=true)

#### After Predicting output for unknown/test data where output is Not Diabetes

![alt text](/DiabetesPrediction_react_flask_docker_aws/images/pic3.png?raw=true)

#### Input the values

![alt text](/DiabetesPrediction_react_flask_docker_aws/images/pic4.png?raw=true)

#### After Predicting output for unknown/test data where output is Diabetes

![alt text](/DiabetesPrediction_react_flask_docker_aws/images/pic5.png?raw=true)
