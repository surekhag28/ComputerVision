import React from 'react';
import {Button} from 'react-bootstrap'
import axios from 'axios';

import  './main.css'

class Main extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      file: null,
      selectedFile : null,
      category: ''
    };

    this.handleUploadImage = this.handleUploadImage.bind(this);
  }

  onFileChange = (event) => {
      event.preventDefault();
      this.setState({file:event.target.files[0],
        selectedFile:URL.createObjectURL(event.target.files[0])
      })
  }

  handleUploadImage = (event) => {
    event.preventDefault();

    const data = new FormData();
    data.append('file',this.state.file)

    axios({
      method: 'post',
      url: 'http://localhost:5000/upload',
      data: data,
      config: { headers: { 'Content-Type': 'multipart/form-data' } }
    })
      .then(response => this.setState({category:response.data.class_name}))
      .catch(errors => console.log(errors))
  }

  render() {
    return (
      <div className="formAlign">
          <form onSubmit={this.handleUploadImage}>
            <h3>Fashion MNIST Classifier</h3>
            <div className="margin-gap">
              <input variant="outline-dark" type="file" onChange={this.onFileChange}/>
              <Button type="submit" variant="success">Predict</Button>
            </div>
            <div >
              <img src={this.state.selectedFile}  className="image-align" alt="img" />
              <p className="category-align">Category is:- {this.state.category}</p>
            </div>
            <div>

            </div>
          </form>
      </div>

    );
  }
}

export default Main;
