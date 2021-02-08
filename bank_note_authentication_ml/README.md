Bank note authentication using Machine Learning and deploying it using Flask web app on Google Cloud Platform as Docker Image.

Steps to be followed for building the end to end pipeline.

1. Create the classifier using any Machine Learning library and train it locally.
2. Save the trained model as pickle file on local storage.
3. Create the web app using Flask framework exposing two REST endpoints to user.
4. In order to view the prediction output of two services, we can use Flasgger API (Swagger).
5. Login to AWS EC2 instance and install Docker if not present.
6. Build the image/ Containeraise the web app using Docker on logged in EC2 instance.
    Command:- docker run -d -p 8000:8000 bank-auth
    
7. Either run the app using "curl" command as shown in snapshot or directly access the Public DNS IP address for running the service.

    Using curl command:-

    ![alt text](/bank_note_authentication_ml/images/pic1.png?raw=true)
    
    Accessing as HTTP service:-
    
    ![alt text](/bank_note_authentication_ml/images/pic2.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic3.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic4.png?raw=true)
    
    ![alt text](/bank_note_authentication_ml/images/pic5.png?raw=true)





 
