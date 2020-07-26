import React from 'react';
import { Table, Tag } from 'antd';

export default class NamedEntityRec extends React.Component {

  constructor(props) {
    super(props);
    this.state = { content: '', response: undefined };
    }

    handleChange = (event) => {
      this.setState({[event.target.name]: event.target.value});
    }

    modifOutput = (out) =>{
      const columns = [
        {
          title: 'Word',
          dataIndex: 'word',
          key: 'word',
        },
        {
          title: 'Pads',
          key: 'pad',
          dataIndex: 'tags',
          render: tags => (
            <React.Fragment> 
              {tags.map(tag => {
                let color = 'geekblue' ;
                if (tag == 'noLabel') {
                  color = 'volcano';
                }
                return (
                  <Tag color={color} key={tag}>
                    {tag.toUpperCase()}
                  </Tag>
                );
              })}
            </React.Fragment>
          ),
        },
      ];

      const data = [

      ];

      out = out.replace(/(?:=){2,}/g, '|').split('\n').map((el) => el.replace(/\s/g,'')).filter((el) => el !== "" && el !== "|").slice(1);
      out.forEach((el,i) => {
        var new_el = {
          key:i,
          word: el.split('|')[0],
          tags: [el.split('|')[1]]
        }
        data.push(new_el);
      });
      return(
        <Table columns={columns} dataSource={data} style={{all: 'initial'}} />
      )

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
          this.setState({response: json});
        });
  
      event.preventDefault();
  }
  returnOut(){
    if (this.state.response){
      var response = this.modifOutput(this.state.response);
      return(
        <section className="container" id="output">
          <div className="container">
              <h2 className="text-center mt-0">Tagged text</h2>
              <hr className="divider my-4" />
              <div className="row justify-content-center">
                  <div className="col-lg-8 text-center">
                  <output >
                          { response }
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
        <section className="container-form" id="form">
              <div className="container">
                  <h2 className="text-center mt-0">Paste a piece of news</h2>
                  <hr className="divider my-4" />
                  <div className="row justify-content-center">
                      <div className="col-lg-8 text-center">
                        <form onSubmit={this.handleSubmit}>
                              <textarea id="text" name="content" placeholder="Text to process ..." onChange={this.handleChange}></textarea>
                          
                              <input className="btn btn-primary btn-xl js-scroll-trigger" type="submit" value="Submit"/>
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
