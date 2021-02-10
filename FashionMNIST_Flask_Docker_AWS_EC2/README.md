### FashionMNIST classifier using Deep Learning (PyTorch framework) and deploying it using Flask web app on AWS EC2 as a Docker Image.

#### Steps to be followed for building the end to end pipeline.

1. Create the classifier using any Deep Learning framework (PyTorch) and train it either locally or on any cloud platform.
2. Save the trained model on local storage.
3. Create the web app using Flask framework exposing REST endpoints to user.
4. In order to view the prediction output of the services, we have created "index.html" file and saved it under templates folder.
5. Login to AWS and launch any free tier EC2 instance.

6. Use ssh service in order to connect to launched remote EC2 instance. <br />
    Command:- ssh -i "image-classifier-key.pem" ec2-user@ec2-13-58-107-101.us-east-2.compute.amazonaws.com <br />
              ssh -i "keypair-file" "Public DNS IP"
    

7. Then install Docker on the instance if not present. <br />
    Command:- sudo yum install docker
    
8. Build the image/ Containeraise the web app using Docker on logged in EC2 instance. <br />
    Command:- docker build -t fashion .
    
9. Run the built docker image to access the service. <br />
    Command:- docker run -d -p 8000:8000 fashion
    
9. Perform inference on the trained model by using HTTP protocol.

    
    ##### Accessing as HTTP service:-
    
    ![alt text](/FashionMNIST_Flask_Docker_AWS_EC2/images/pic1.png?raw=true)
    
     


 
