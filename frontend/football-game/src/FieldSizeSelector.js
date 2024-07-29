import React, { useState } from 'react';
import axios from 'axios';
import './FieldSizeSelector.css';

const FieldSizeSelector = ({ onSizeChange }) => {
  const [width, setWidth] = useState(13);
  const [height, setHeight] = useState(9);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/set_field_size', { width, height });
      onSizeChange(response.data);
    } catch (error) {
      console.error('Error setting field size:', error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="field-size-form">
      <div className="input-group">
        <label htmlFor="width">Width:</label>
        <input
          id="width"
          type="number"
          value={width}
          onChange={(e) => setWidth(Number(e.target.value))}
          min="7"
          max="20"
        />
      </div>
      <div className="input-group">
        <label htmlFor="height">Height:</label>
        <input
          id="height"
          type="number"
          value={height}
          onChange={(e) => setHeight(Number(e.target.value))}
          min="5"
          max="15"
        />
      </div>
      <button type="submit" className="submit-button">Set Field Size</button>
    </form>
  );
};

export default FieldSizeSelector;
