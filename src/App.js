import React from 'react';

import './App.css';
import Car from './Component/Car/Car';
import Scooter from './Component/Scooter/Scooter';

function App() {
  return (
    <div className="App">
       <h1> hello world </h1>
       <div>
        <Car/>
        <Scooter/>
       </div>
    </div>
  );
}

export default App;
