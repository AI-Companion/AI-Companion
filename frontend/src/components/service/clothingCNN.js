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
  

    handleSubmit = (event) => {
      const { content } = this.state;
      fetch('/api/clothingClassifier', {
          method: 'POST',
          body: JSON.stringify({content : this.state.content}),
          headers: {
            'Content-Type': 'application/json',
          },
        }).then(function(response) {
          return response.json();
        }).then((json)=>{
          console.log("json",json)
          this.setState({response: json});
        });
  
      event.preventDefault();
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
          Hi this is rendering for clothing CNN
      </div>
    );  
  }
}