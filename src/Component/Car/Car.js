import React, { useEffect, useState } from 'react';

function Car() {
  const [student, setStudent] = useState({});

  useEffect(() => {
    fetch("http://localhost:3001/api/student")
      .then((res) =>res.json() )
      .then((data) => setStudent(data.data[0])
      );
  }, []);
  return (
    <div className="Car">
       <div className='table'>
       <table>
        <thead>
          <tr>
            <th>student name</th>
            <th>age</th>
           
          </tr>
        </thead>
        <tbody>.
          <tr>
            <td>{student.name}</td>
            <td>{student.age}</td>
            
          </tr>
        
        </tbody>
      </table>
       </div>
    </div>
  );
}

export default Car;
