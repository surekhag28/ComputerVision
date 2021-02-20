Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@surekhag28 
Learn Git and GitHub without any code!
Using the Hello World guide, you’ll start a branch, write comments, and open a pull request.


surekhag28
/
ComputerVision
1
00
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
ComputerVision
/
fashionMNIST-react-flask-docker
/
README.md
in
main
 

Spaces

11

Soft wrap
1
# Fashion MNIST Classifier - Front end: React, REST endpoints: Flask web app, Container: Docker, Deployment: AWS EC2
2
​
3
## Overview
4
As part of this sample project, I have created simple webapplication which will take input as image from the user and will do the inference at backend using pretrained deep learning model (fashionMNIST classifier). As a result of inference, it will send the category or class name of the image and finally output will be displayed to the user.
5
​
6
### Docker Compose
7
Docker maintains software and all of its dependencies within a "container", which can make collaborating and deploying simpler. 
8
Docker Compose is a tool for easily managing applications running multiple Docker containers.
9
​
10
Since the project involves development of two different applications, front end using React and  backend using Flask web app, we will need docker compose to containerise them and then finally deploying on AWS EC2 instance.
11
​
12
### Front end -- React App
13
At the front end side we are taking input as image from the user and allows him/her to verify the category of the image by hitting 'Predict" button.
14
​
15
### Back end -- Flask web app
16
Request from the user containing image as form data will be sent to concerned REST end point.
17
The service will upload the image on server and will do the inference using pretrained deep learning model (fashionMNIST classifier).
18
As a result it will send category details in json format to react app which will display the class label to the user.
19
​
20
### Deployment -- AWS EC2
21
Both the applications are dockerised using docker-compose and finally deployed on AWS EC2 instance. <br /><br />
22
​
23
***Note :- 1. Make sure to change the URLs with the Public DNS address in the application while sending request to the Flask web app. <br />
24
           2. Also ensure to implement the cross origin policy in the web app otherwise request from the react application will be blocked.***
25
​
26
#### Before sending the request to the server
27
​
28
![alt text](/fashionMNIST-react-flask-docker/images/pic1.png?raw=true)
29
​
30
#### After getting the inference output from the backend server
31
​
32
![alt text](/fashionMNIST-react-flask-docker/images/pic2.png?raw=true)
33
​
34
​
35
​
@surekhag28
Commit changes
Commit summary
Create README.md
Optional extended description
Add an optional extended description…
 Commit directly to the main branch.
 Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
