import React from 'react';
import {
    Link
  } from "react-router-dom";
import NamedEntityRec from './service/NamedEntityRec';
import SentimentAnalysis from './service/SentimentAnalysis';
import ClothingCNN from './service/clothingCNN';

export default class Service extends React.Component {
    
    constructor(props) {
        super(props);
        }

  render() {
    let ret;
    if (this.props.serv == "namedEntity")
        ret = <NamedEntityRec/>
    else if (this.props.serv == "clothingService")
        ret = <ClothingCNN/>
    else
        ret = <SentimentAnalysis/>
    return (
        <div>
            <nav className="navbar navbar-expand-lg navbar-light fixed-top py-3" id="mainNav" >
            <Link className="navbar-brand js-scroll-trigger" to="/" style={{color: 'black'}}>
            Ai-Companion
            </Link>
            </nav>
            <br/>
                <div className="container">
                        <button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                            <span className="navbar-toggler-icon"></span>
                        </button>
                    <div className="collapse navbar-collapse" id="navbarResponsive"/>
                    {ret}
                </div>
        </div>
    );  
  }
}