import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import MasterHead from './components/MasterHead';
import Navigation from './components/Navigation';
import Portfolio from'./components/Portfolio';
import Services from './components/Services';
import Service from './components/Service';
import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";

class App extends Component {
  render() {
    return (
      <Router>
        <div className="App">
        <Switch>
            <Route path="/sentimentAnalysis">
              <Service serv=""/>
            </Route>
            <Route path="/namedEntityRecognition">
              <Service serv="namedEntity"/>
            </Route>
            <Route path="/clothingClassifier">
              <Service serv="clothingService"/>
            </Route>
            <Route exact={true} path="/">
              <MasterHead/>
              <Navigation/>
              <Portfolio/>
              <Services/>
            </Route>
          </Switch>
      </div>
      </Router>
    );
  }
}

export default App;
