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
      m:0.0,
      c:0.0,
      x_val:0.0,
      y_pred:0.0,
      grad_img:null,
      cost_img:null,
      img_stat:0
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
      console.log(response.data)

      this.setState({rmse:response.data.root_mean_square_error.toFixed(2),
                      r_squared:response.data.r_squared.toFixed(2),
                      m:response.data.m.toFixed(2),
                      c:response.data.c.toFixed(2),
                      grad_img:response.data.grad_img,
                      cost_img:response.data.cost_img,
                      img_stat:1})
    }).catch(error => console.log(error))
  }

  handlePrediction = (event) => {
    event.preventDefault();
    
    let pred = this.state.x_val*this.state.m+this.state.c
    this.setState({y_pred : parseFloat(pred).toFixed(2)});
  }

  render(){
    let img_comp;
    
    if(this.state.img_stat===1){
      console.log('yes image')
      img_comp = <div class="row">
                  <div class="column">
                    <img src={`data:image/jpeg;base64,${this.state.grad_img}`} className='imageAlign1'/>
                  </div>
                  <div class="column">
                    <img src={`data:image/jpeg;base64,${this.state.cost_img}`} className='imageAlign2'/>
                  </div>
                </div>
    }

    return(
      <div>
        <Card className="bg-dark text-white text-center cardAlign">
          <Card.Header>Linear Regression using Gradient Descent</Card.Header>
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
                <p>Coefficient m :- {this.state.m}</p>
                <p>Coefficient c :- {this.state.c}</p>
                {img_comp}
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
