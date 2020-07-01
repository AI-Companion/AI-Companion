import React from 'react';
import ReactDOM from 'react-dom';

export default class NamedEntityRec extends React.Component {

  constructor(props) {
    super(props);
    this.state = { content: '', response: undefined };
    }

    handleChange = (event) => {
      this.setState({[event.target.name]: event.target.value});
    }
  

    handleSubmit = (event) => {
      const { content } = this.state;
      fetch('/api/namedEntityRecognition', {
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
        <section class="container-form" id="form">
              <div class="container">
                  <h2 class="text-center mt-0">Paste a piece of news</h2>
                  <hr class="divider my-4" />
                  <div class="row justify-content-center">
                      <div class="col-lg-8 text-center">
                        <form onSubmit={this.handleSubmit}>
                              <textarea id="text" name="content" placeholder="Text to process ..." onChange={this.handleChange}></textarea>
                          
                              <input class="btn btn-primary btn-xl js-scroll-trigger" type="submit" value="Submit"/>
                        </form>
                      </div>
                  </div>
              </div>
        </section>
        {out}
      </div>
    );  
  }
}