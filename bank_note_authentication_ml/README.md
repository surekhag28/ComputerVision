### Bank note authentication using Machine Learning and deploying it using Flask web app on Google Cloud Platform as Docker Image.

#### Steps to be followed for building the end to end pipeline.

1. Create the classifier using any Machine Learning library and train it locally.
2. Save the trained model as pickle file on local storage.
3. Create the web app using Flask framework exposing two REST endpoints to user.
4. In order to view the prediction output of two services, we can use Flasgger API (Swagger).
5. Login to AWS and launch any free tier EC2 instance.

6. Use ssh service in order to connect to launched remote EC2 instance. <br />
    Command:- ssh -i "bank-auth-key.pem" ec2-user@ec2-13-58-107-101.us-east-2.compute.amazonaws.com <br />
              ssh -i <keypair-file> <Public DNS IP>
    

7. Then install Docker on the instance if not present. <br />
    Command:- sudo yum install docker
    
8. Build the image/ Containeraise the web app using Docker on logged in EC2 instance. <br />
    Command:- docker build -t bank_note_auth .
    
9. Run the built docker image to access the service. <br />
    Command:- docker run -d -p 8000:8000 bank-auth
    
9. Either run the app using "curl" command as shown in snapshot or directly access the Public DNS IP address for running the service.

    ##### Using curl command:-

    ![alt text](/bank_note_authentication_ml/images/pic1.png?raw=true)
    
    ##### Accessing as HTTP service:-
    
    ![alt text](/bank_note_authentication_ml/images/pic2.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic3.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic4.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic5.png?raw=true)





 
