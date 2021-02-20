As part of this sample project, I have created simple webapplication which will take input as image from the user and will do the inference at backend using pretrained deep learning model (fashionMNIST classifier). As a result of inference, it will send the category or class name of the image and finally output will be displayed to the user.

### Docker Compose
Docker maintains software and all of its dependencies within a "container", which can make collaborating and deploying simpler. 
Docker Compose is a tool for easily managing applications running multiple Docker containers.

Since the project involves development of two different applications, front end using React and  backend using Flask web app, we will need docker compose to containerise them and then finally deploying on AWS EC2 instance.

### Front end -- React App
At the front end side we are taking input as image from the user and allows him/her to verify the category of the image by hitting 'Predict" button.

### Back end -- Flask web app
Request from the user containing image as form data will be sent to concerned REST end point.
The service will upload the image on server and will do the inference using pretrained deep learning model (fashionMNIST classifier).
As a result it will send category details in json format to react app which will display the class label to the user.

### Deployment -- AWS EC2
Both the applications are dockerised using docker-compose and finally deployed on AWS EC2 instance. <br /><br />

***Note :- 1. Make sure to change the URLs with the Public DNS address in the application while sending request to the Flask web app. <br />
           2. Also ensure to implement the cross origin policy in the web app otherwise request from the react application will be blocked.***

#### Before sending the request to the server

![alt text](/fashionMNIST-react-flask-docker/images/pic1.png?raw=true)

#### After getting the inference output from the backend server

![alt text](/fashionMNIST-react-flask-docker/images/pic2.png?raw=true)
