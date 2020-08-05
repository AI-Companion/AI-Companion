import React from 'react';
import axios from 'axios';
export default class ClothingCNN extends React.Component {

  constructor(props) {
    super(props);
    this.state = {response: undefined };
    }
  

  handleChange = (event) => {
    this.setState({[event.target.name]: event.target.value});
  }
    
  state = {
    title: '',
    content: '',
    image: null
  };

  handleImageChange = (e) => {
    var image = document.getElementById('output');
    image.src = URL.createObjectURL(e.target.files[0]);
    this.setState({
      image: e.target.files[0]
    })
    this.setState({response: undefined})
  };

  handleSubmit = (e) => {
    e.preventDefault();
    let form_data = new FormData();
    form_data.append('image', this.state.image, this.state.image.name);
    let url = '/api/clothingClassifier';
    axios.post(url, form_data, {
      headers: {
        'content-type': 'multipart/form-data'
      }
    })
        .then(res => {
          this.setState({response: res.data})
        })
        .catch(err => console.log(err))
  };

  render() {
    if(this.state.response){
    var out = (<div className="col-lg-5 col-sm-7">
            <h1>{this.state.response}</h1>
          </div>);
    };
    return (
      <div>
        <section className="container-form" id="form">
            <div className="container">
                <h2 className="text-center mt-0">Upload an Image to know what it is </h2>
                <hr className="divider my-4" />
                <div className="row justify-content-center">
                    <div className="col-lg-8 text-center">
                      <form onSubmit={this.handleSubmit}>
                        <div className="row no-gutters">
                          <div className="col-lg-8 text-center">
                            <p>
                                <input type="file"
                                    id="image"
                                    accept="image/png, image/jpeg"  onChange={this.handleImageChange} required/>
                            </p>
                            <input type="submit"/>
                          </div>
                        </div>
                      </form>
                    </div>
                </div>
            </div>
        </section>
        <br/><br/>
         <div className="row">
           <div className="col-lg-6 col-sm-10">
             <img alt="" id="output" />	
           </div>
           {out}
            
        </div>
      </div>
    );
  }
}