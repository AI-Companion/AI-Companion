import React from 'react';
import ReactDOM from 'react-dom';

export default class ClothingCNN extends React.Component {

  constructor(props) {
    super(props);
    this.state = { content: '', response: undefined };
    }

    handleChange = (event) => {
      this.setState({[event.target.name]: event.target.value});
    }

    handleSubmit = (data) => {
      fetch('/api/clothingCnn', {
          method: 'POST',
          body: JSON.stringify(data),
          headers: {
            'Content-Type': 'application/json',
          },
        }).then(function(response) {
          return response.json();
        }).then((json)=>{
          console.log("json",json)
          this.setState({response: json});
        });
  
  }
  

  uploadFile = (event) => {
    event.preventDefault();
    let form_data = new FormData();
    form_data.append('content', this.state.content);
    fetch('/api/clothingClassifier', {
      method: 'POST',
      body: form_data,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then(function(response) {
      return response.json();
    }).then((json)=>{
      console.log("json",json)
      this.setState({response: json});
    });

    }

  returnOut(){
    console.log("response", this.state.response)
    if (this.state.response){
      return(
        <section class="container" id="output">
          <div class="container">
              <h2 class="text-center mt-0">Tagged text</h2>
              <hr class="divider my-4" />
              <div class="row justify-content-center">
                  <div class="col-lg-8 text-center">
                  <output >
                          { this.state.response }
                  </output>
                  </div>
              </div>
          </div>
        </section>
      )
    }
    else
    return(<div></div>)
  }

  render() {
    var out = this.returnOut();
    return (
      <div>
        <section class="container-form" id="form">
          <form onSubmit={this.uploadFile}>

            <label>
              File
              <input type="file" name="content" onChange={this.handleChange} />
            </label>

            <input type="submit" value="Upload Image"/>
          </form>
        </section>
        {out}
      </div>
    );  
  }
}