import React,{Component} from 'react'
import {Button,Card} from 'react-bootstrap'
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios'

class App extends Component{
  constructor(props){
    super(props)

    this.state = {
      file:null,
      rmse:0.0,
      r_squared:0.0,
      b0:0.0,
      b1:0.0,
      x_val:0.0,
      y_pred:0.0,
      img_path:null
    }

    this.onFileChange = this.onFileChange.bind(this);
    this.onInputChange = this.onInputChange.bind(this);
    this.handleFileUpload = this.handleFileUpload.bind(this);
    this.handlePrediction = this.handlePrediction.bind(this);
  }

  onFileChange = (event) => {
    event.preventDefault();
    this.setState({file:event.target.files[0]})
  }

  onInputChange = (event) => {
    event.preventDefault();
    this.setState({x_val:parseFloat(event.target.value).toFixed(2)})
  }

  handleFileUpload = (event) => {
    event.preventDefault();

    const formData = new FormData()
    formData.append('file',this.state.file)

    axios({
      method:'post',
      url:'http://ec2-3-22-101-30.us-east-2.compute.amazonaws.com:5000/upload',
      data:formData,
      config:{headers:{'Content-Type':'multipart/form-data'}}
    }).then(response =>{
      this.setState({rmse:response.data.root_mean_sqaure_error.toFixed(2),
                      r_squared:response.data.r_squared.toFixed(2),
                      b0:response.data.b0.toFixed(2),
                      b1:response.data.b1.toFixed(2)})
    }).catch(error => console.log(error))
  }

  handlePrediction = (event) => {
    event.preventDefault();
    
    let pred = this.state.x_val*this.state.b1+this.state.b0
    this.setState({y_pred : parseFloat(pred).toFixed(2)});
  }

  render(){
    return(
      <div>
        <Card className="bg-dark text-white text-center cardAlign">
          <Card.Header>Linear Regression using Ordinary Least Square</Card.Header>
          <Card.Body>
            <Card.Title>Linear Regression</Card.Title>
            <form onSubmit={this.handleFileUpload}>
              <div>
                <input type='file' onChange={this.onFileChange}/>
                <Button type='submit' variant='success'>Train</Button>
              </div>
              <div className='valueDisplay'>
                <p>Root Mean Squared error is:- {this.state.rmse}</p>
                <p>R-Squared score is:- {this.state.r_squared}</p>
                <p>Coefficient b0 :- {this.state.b0}</p>
                <p>Coefficient b1 :- {this.state.b1}</p>
              </div>
            </form>
          </Card.Body>
          <Card.Footer className="text-white">
            <form onSubmit={this.handlePrediction}>
              <div>
                Enter X value:- <input type='text' name='x_val' onChange={this.onInputChange}/>
                <Button type='submit' variant='success'>Predict</Button>
              </div>
              <div className='textAlign'>
                <p>Predicted Y value is :- {this.state.y_pred}</p>
              </div>
            </form>
          </Card.Footer>
        </Card>
        </div>
    )
  }
}

export default App;
