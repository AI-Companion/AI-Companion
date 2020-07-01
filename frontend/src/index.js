import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';

const marabou = React.createContext('localhost:5000');

ReactDOM.render(
  <App />,
  document.getElementById('root')
);
