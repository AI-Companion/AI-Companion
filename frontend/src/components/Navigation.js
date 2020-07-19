import React, {useEffect} from 'react';
import ReactDOM from 'react-dom';
//import 'style.css';
//import masterHead from './components/masterHead';


export default class Navigation extends React.Component {
  constructor(){
    super()
  }
  
  render() {
    return (
        <nav className="navbar navbar-expand-lg navbar-light fixed-top py-3" id="mainNav">
        <div className="container">
            <a className="navbar-brand js-scroll-trigger" href="#page-top">Companion.ai</a><button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span className="navbar-toggler-icon"></span></button>
            <div className="collapse navbar-collapse" id="navbarResponsive">
                <ul className="navbar-nav ml-auto my-2 my-lg-0">
                    <li className="nav-item"><a className="nav-link js-scroll-trigger" href="#portfolio">Our Services</a></li>
                    <li className="nav-item"><a className="nav-link js-scroll-trigger" href="#contact">Contact</a></li>
                </ul>
            </div>
        </div>
    </nav>
    );
  }
}