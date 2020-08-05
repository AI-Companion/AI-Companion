import React from 'react';
import Smiling_Face from '../../assets/img/portfolio/thumbnails/Smiling_Face.png'
import Angry_Face from '../../assets/img/portfolio/thumbnails/Angry_Face.png'

export default class SentimentAnalysis extends React.Component {

  constructor(props) {
    super(props);
    this.state = { content: '', response: undefined };
    }

    handleChange = (event) => {
      this.setState({[event.target.name]: event.target.value});
    }
  

    handleSubmit = (event) => {
      fetch('/api/sentimentAnalysis', {
          method: 'POST',
          body: JSON.stringify({content : this.state.content}),
          headers: {
            'Content-Type': 'application/json',
          },
        }).then(function(response) {
          return response.json();
        }).then((json)=>{
          this.setState({response: json});
        });
  
      event.preventDefault();
  }
  returnOut(){
    if (this.state.response){
      if (this.state.response >= 50){
        return(
              <div className="row">
                <div className="col-lg-4 col-sm-6">
                    <img className="img-fluid" src={Smiling_Face} alt="Happy Sentiment" width="100" height="100"/>
                </div>
                <div className="col-lg-4 col-sm-6">
                    <output ><h2><b>% { (this.state.response).toFixed(2) }</b></h2></output>
                </div>
              </div>
        )
      }
      else{
        return(
        <div>
          <div className="col-lg-4 col-sm-6">
              <img className="img-fluid" src={Angry_Face} alt="Angry Sentiment" width="100" height="100"/>
          </div>
          <div className="col-lg-4 col-sm-6">
              <output ><h2>%<b> { (100 - this.state.response).toFixed(2) }</b></h2></output>
          </div>
        </div>
        )
      }
    }
    else
    return(<div></div>)
  }

  render() {
    var out = this.returnOut();
    return( 
      <div>
        <section className="container-form" id="form">
            <div className="container">
                <h2 className="text-center mt-0">Write your review about the last movie you watched</h2>
                <hr className="divider my-4" />
                <div className="row justify-content-center">
                    <div className="col-lg-8 text-center">
                        <form onSubmit={this.handleSubmit}>
                            <textarea id="text" name="content" placeholder="Text to process ..." onChange={this.handleChange} ></textarea>
                        
                            <input className="btn btn-primary btn-xl js-scroll-trigger" type="submit" value="Submit"/>
                        </form>
                    </div>
                </div>
            </div>
        </section>
        <section id="output">
            <div className="container-fluid p-0">
               {out}
            </div>
        </section>
      </div>
    );  
  }
}