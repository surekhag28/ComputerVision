import React,{Component} from 'react';
import {Button} from 'react-bootstrap';
import axios from 'axios';
import Result from './Result.jsx'

class DataForm extends Component {
    constructor(props){
        super(props)

        this.state = {
            pregnancies: 0,
            glucose: 0,
            bloodpressure: 0,
            skinthickness: 0,
            insulin: 0,
            bmi: 0.0,
            dpf: 0.0,
            age: 0,
            pred:''
        }


        this.inputChange = this.inputChange.bind(this);
        this.handlePredict = this.handlePredict.bind(this);
    }

    handlePredict = (event) => {
        event.preventDefault();

        const data = new FormData();
        data.append('pregnancies',this.state.pregnancies);
        data.append('glucose',this.state.glucose);
        data.append('bloodpressure',this.state.bloodpressure);
        data.append('skinthickness',this.state.skinthickness);
        data.append('insulin',this.state.insulin);
        data.append('bmi',this.state.bmi);
        data.append('dpf',this.state.dpf);
        data.append('age',this.state.age);

        axios({
            method:'post',
            url:'http://ec2-3-129-211-233.us-east-2.compute.amazonaws.com:5000/predict',
            data:data,
            config:{headers:{'Content-Type':'multipart/form-data'}}
        }).then(response => {
            console.log(response)
            this.setState({pred:response.data.class})
        }).catch(error => console.log(error))
    }

    inputChange = (event) => {
        event.preventDefault();
        this.setState({
            [event.target.name]: event.target.value
        })
    }

    render(){
        if(this.state.pred !==''){
            return <Result pred={this.state.pred}/>
        }
        return(
            <form onSubmit={this.handlePredict}>
                <input className='form-input formInput' type='text' name='pregnancies' placeholder='Number of Pregnancies eg. 0' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='glucose' placeholder='Glucose (mg/dL) eg. 80' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='bloodpressure' placeholder='Blood Pressure (mmHg) eg. 80' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='skinthickness' placeholder='Skin Thickness (mm) eg. 20' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='insulin' placeholder='Insulin Level (IU/mL) eg. 80' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='bmi' placeholder='Body Mass Index (kg/m²) eg. 23.1' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='dpf' placeholder='Diabetes Pedigree Function eg. 0.52' onChange={this.inputChange}/><br/>
                <input className='form-input formInput' type='text' name='age' placeholder='Age (years) eg. 34'/><br/>
                <Button type='submit' variant="outline-success">Predict</Button>{' '}
            </form>
        )
    }
}

export default DataForm;
