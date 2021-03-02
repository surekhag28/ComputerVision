import React from 'react'
import {Card} from 'react-bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import DataForm from './DataForm.jsx'

function App() {
  return(
    <div>
      <Card className="text-center">
        <Card.Header className='bg-dark text-white'>Diabetes Predictor</Card.Header>
        <Card.Body>
          <Card.Title>A Machine Learning App.</Card.Title>
          <Card.Text>
            Built with Flask,React and deployed as dockerised image on AWS EC2
          </Card.Text>
          <DataForm/>
        </Card.Body>
        <Card.Footer className='bg-dark text-white'>
            <div className="contact">
                <a href="https://github.com/anujvyas/Diabetes-Prediction-Deployment"><i className="fab fa-github fa-lg contact-icon"></i></a>&nbsp;&nbsp;
                <a href="https://www.linkedin.com/in/anujkvyas"><i className="fab fa-linkedin fa-lg contact-icon"></i></a>
            </div>
            <p className='footer-description'>Made by Surekha Gaikwad.</p>
        </Card.Footer>
      </Card>
    </div>
  )
}

export default App;
