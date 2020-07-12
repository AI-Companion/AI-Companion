import React from 'react';
import ReactDOM from 'react-dom';

export default class Services extends React.Component {
  render() {
    return (
        <section class="page-section" id="services">
        <div class="container">
            <h2 class="text-center mt-0">At Your Service</h2>
            <hr class="divider my-4" />
            <div class="row">
                <div class="col-lg-3 col-md-6 text-center">
                    <div class="mt-5">
                        <i class="fas fa-4x fa-gem text-primary mb-4"></i>
                        <h3 class="h4 mb-2">Made with innovation</h3>
                        <p class="text-muted mb-0">We work with you to deliver actionable ariticial intelligence products</p>
                        <p class="text-muted mb-0">Our solutions consider users, business value and programming standards at the heart of our development strategies</p>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 text-center">
                    <div class="mt-5">
                        <i class="fas fa-4x fa-laptop-code text-primary mb-4"></i>
                        <h3 class="h4 mb-2">Made with ease</h3>
                        <p class="text-muted mb-0">Our solutions are designed to be easy to deploy, use and maintain</p>
                        <p class="text-muted mb-0">You will not need a big machine learning crew to keep uptodate with state of the art artificial intelligence</p>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 text-center">
                    <div class="mt-5">
                        <i class="fas fa-4x fa-heart text-primary mb-4"></i>
                        <h3 class="h4 mb-2">Made with love</h3>
                        <p class="text-muted mb-0">Is it really open source if it's not made with love?</p>
                        <p class="text-muted mb-0">Please visit our github repository</p>
                        <p class="text-muted mb-0">https://github.com/mmarouen/marabou</p>
                    </div>
                </div>
                <div class="col-lg-3 col-md-6 text-center">
                    <div class="mt-5">
                        <i class="fas fa-4x fa-graduation-cap text-primary mb-4"></i>
                        <h3 class="h4 mb-2">Made with care</h3>
                        <p class="text-muted mb-0">Make use of our extensive professionnal experience</p>
                        <p class="text-muted mb-0">We specialize in improving your operations and provide new services!</p>
                    </div>
                </div>
            </div>
        </div>
        </section>
    );
  }
}