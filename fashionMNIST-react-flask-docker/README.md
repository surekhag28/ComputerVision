# Fashion MNIST Classifier - Front end: React, REST endpoints: Flask web app, Container: Docker, Deployment: AWS EC2

## Overview
As part of this sample project, I have created simple webapplication which will take input as image from the user and will do the inference at backend using pretrained deep learning model (fashionMNIST classifier). As a result of inference, it will send the category or class name of the image and finally output will be displayed to the user.

### Why Create React App?
[Create React App](https://facebook.github.io/create-react-app/) allows 
us to very easily *create a React app* with no build configuration. React is 
currently one of the most popular front-end Javascript libraries for 
building UIs.

### Why Flask?
Flask is a lightweight, highly-customizable micro-framework for Python. It let's
us build really simple web applications quickly ([the "hello world" app is literally 5 
lines of code](http://flask.pocoo.org/docs/1.0/quickstart/#a-minimal-application)).
Flask doesn't come built-in with much, and if you're looking to integrate a more 
robust back-end framework with React (say, Ruby on Rails), I'd recommend checking
out [this blog post](https://medium.com/superhighfives/a-top-shelf-web-stack-rails-5-api-activeadmin-create-react-app-de5481b7ec0b).

### Why Docker Compose?
[Docker](https://www.docker.com/) maintains software and all of its dependencies within a "container",
which can make collaborating and deploying simpler. [Docker Compose](https://docs.docker.com/compose/)
is a tool for easily managing applications running multiple Docker containers. 

## How to Use
Firstly, download [Docker desktop](https://www.docker.com/products/docker-desktop) and follow its
 instructions to install it. This allows us to start using Docker containers.
 
Create a local copy of this repository and run

    docker-compose build
    
This spins up Compose and builds a local development environment according to 
our specifications in [docker-compose.yml](docker-compose.yml). Keep in mind that 
this file contains settings for *development*, and not *production*.

After the containers have been built (this may take a few minutes), run

    docker-compose up
    
This one command boots up a local server for Flask (on port 5000)
and React (on port 3000). Head over to

    http://localhost:3000/ 
    
to view an incredibly underwhelming React webpage listing two fruits and their
respective prices. 
Though the apparent result is underwhelming, this data was retrieved through an API call
 to our Flask server, which can be accessed at

    http://localhost:5000/api/v1.0/test
    
The trailing '*/api/v1.0/test*' is simply for looks, and can be tweaked easily
in [api/app.py](api/app.py). The front-end logic for consuming our API is
contained in [client/src/index.js](client/src/index.js). The code contained within
these files simply exists to demonstrate how our front-end might consume our back-end
API.

Finally, to gracefully stop running our local servers, you can run
 
    docker-compose down

in a separate terminal window or press __control + C__.
