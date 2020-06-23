import React, {useEffect} from 'react';
import ReactDOM from 'react-dom';
import 'style.css';


class App extends React.Component {
  render() {
    return (
      <div className="App">
        <h1> Hello BriteCore </h1>
        <p> Here is your app </p>
      </div>
    );
  }
}

ReactDOM.render(
  <App />,
  document.getElementById('root')
);
