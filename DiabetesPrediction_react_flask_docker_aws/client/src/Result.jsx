import React,{Component} from 'react'
import Diabetes from './assets/images/diabetes.gif';
import NoDiabetes from './assets/images/no-diabetes.gif'

class Result extends Component {
    constructor(props){
        super(props)        
    }

    renderElement = (pred) => {
        if(pred==='0'){
            console.log(pred)
            return(
                <div>
                    <h1>Prediction: <span className='safe'>Hurray! you do not have DIABETES.</span></h1>
               	    <img className='gif' src={NoDiabetes} alt='No diabetes image'/> 
		</div>        
            )
        }else{
            console.log(pred)
            return(
                <div>
                    <h1>Prediction: <span className='danger'>Its sad, but you have DIABETES and need medication.</span></h1>
                    <img className='gif1' src={Diabetes} alt='Diabetes image'/>
		</div>        
            )
        }
    }

    render(){
        return(
            <div>
                {this.renderElement(this.props.pred)}
            </div>
        )
    }
}

export default Result
