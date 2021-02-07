Bank note authentication using Machine Learning and deploying it using Flask web app on Google Cloud Platform as Docker Image.

Steps followed for building the end to end pipeline.

1. Created classifier using Machine Learning library "scikit-learn".
    In order to train the model on given dataset, I used RandomForest classifier from sklearn ML library which end up giving accuracy score as 0.98.
  
2. Saved the trained model locally as pickle file.
    After traing the model, saved it as pickle file on local system in order to use it later for inference purpose from web app.
    
3. Created web app using Flask.
    Now, In order to access the model for inference purpose, I created two end points using Flask mini web-framework. One end point takes data as query parameter from user while in the other we are taking number of observations from file (csv file).
    
 4. Testing end points using Postman
 After building the endpoints, they can be easily tested using Postman Rest client or using "curl" command.
 
 5. Exposing model inference using front-end application
 The endpoints can also be accessed via front end application which is developed using Flasgger API. Flasgger is a Flask extension to extract OpenAPI-Specification from all Flask views registered in our API. It also comes with SwaggerUI embedded so that we can access http://localhost:5000/apidocs and visualize and interact with the API resources.
 
 6. Containerasiation using Docker
 Created the docker image for the application as per the commands specified in Dockerfile.
 
 7. Deployment on Google Kubernetes engine cloud.
 
