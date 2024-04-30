import React, { useEffect, useState } from 'react';

function Car() {
  const [students, setStudents] = useState([]);

  useEffect(() => {
    fetch("http://localhost:3001/api/student")
      .then((res) => res.json())
      .then((data) => {
        setStudents(data.data);
      });
  }, []);

  return (
    <div className="Car">
      <div className='table'>
        <table>
          <thead>
            <tr>
              <th>Student name</th>
              <th>Age</th>
            </tr>
          </thead>
          <tbody>
            {students.map((student, index) => (
              <tr key={index}>
                <td>{student.name}</td>
                <td>{student.age}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default Car;
