import React, {useEffect, Component} from 'react';
import ReactDOM from 'react-dom';

class MasterHead extends React.Component {

  render() {
    return (
        <header className="masthead">
        <div className="container h-100">
            <div className="row h-100 align-items-center justify-content-center text-center">
                <div className="col-lg-10 align-self-end">
                    <h1 className="text-uppercase text-white font-weight-bold">We help you brige the AI gap for your business</h1>
                    <hr className="divider my-4" />
                </div>
                <div className="col-lg-8 align-self-baseline">
                <p className="text-white-75 font-weight-bold mb-5">We are a software programming company specializing in artificial intelligence services</p>
                <p className="text-white-75 font-weight-bold mb-5">Our mission is to deliver AI driven solution for your business to better know your customers, innovate, increase revenue and reduce costs.
Thanks to the AI companion, you give your business the chance to succeed through the use of big data and artificial intelligence technologies.</p>
                    <a className="btn btn-primary btn-xl js-scroll-trigger" href="#portfolio">Find Out More</a>
                </div>
            </div>
        </div>
    </header>
    );
  }
}
export default MasterHead;