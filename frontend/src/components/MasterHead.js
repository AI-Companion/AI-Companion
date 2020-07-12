import React, {useEffect, Component} from 'react';
import ReactDOM from 'react-dom';

class MasterHead extends React.Component {

  render() {
    return (
        <header class="masthead">
        <div class="container h-100">
            <div class="row h-100 align-items-center justify-content-center text-center">
                <div class="col-lg-10 align-self-end">
                    <h1 class="text-uppercase text-white font-weight-bold">What is natural language processing?</h1>
                    <hr class="divider my-4" />
                </div>
                <div class="col-lg-8 align-self-baseline">
                    <p class="text-white-75 font-weight-light mb-5">Marabou is designed to provide a deep analysis of textual content for non technical users. With a few clicks you can extract the emotional tone hidden behind tons of expressions in your data or locate and classify named entity present in your data such as geographical locations, organizations ... have fun!</p>
                    <a class="btn btn-primary btn-xl js-scroll-trigger" href="#portfolio">Find Out More</a>
                </div>
            </div>
        </div>
    </header>
    );
  }
}
export default MasterHead;