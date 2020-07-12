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
    console.log("here we go again ", this.props.serv)
    if (this.props.serv == "namedEntity")
        ret = <NamedEntityRec/>
    else if (this.props.serv == "")
        ret = <SentimentAnalysis/>
    else
        ret = <ClothingCNN/>
    return (
        <div>
            <nav class="navbar navbar-expand-lg navbar-light fixed-top py-3" id="mainNav" >
            <Link class="navbar-brand js-scroll-trigger" to="/" style={{color: 'black'}}>
                Companion.ai
            </Link>
            </nav>
            <br/>
                <div class="container">
                        <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                            <span class="navbar-toggler-icon"></span>
                        </button>
                    <div class="collapse navbar-collapse" id="navbarResponsive"/>
                    {ret}
                </div>
        </div>
    );  
  }
}