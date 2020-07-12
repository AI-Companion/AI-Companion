import React, {useEffect, Component} from 'react';
import ReactDOM from 'react-dom';

class MasterHead extends React.Component {

  render() {
    return (
        <header class="masthead">
        <div class="container h-100">
            <div class="row h-100 align-items-center justify-content-center text-center">
                <div class="col-lg-10 align-self-end">
                    <h1 class="text-uppercase text-white font-weight-bold">We help you brige the AI gap for your business</h1>
                    <hr class="divider my-4" />
                </div>
                <div class="col-lg-8 align-self-baseline">
                    <p class="text-white-75 font-weight-bold mb-5">Get AI driven solution for your business to better know your customers, innovate, increase revenue and reduce costs.
Thanks to the AI companion, you give your business the chance to succeed through the use of big data and artificial intelligence technologies.</p>
                    <a class="btn btn-primary btn-xl js-scroll-trigger" href="#portfolio">Find Out More</a>
                </div>
            </div>
        </div>
    </header>
    );
  }
}
export default MasterHead;