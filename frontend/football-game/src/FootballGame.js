import React, { useState, useEffect } from 'react';
import axios from 'axios';
import FieldSizeSelector from './FieldSizeSelector';
import './FootballGame.css';

const FootballGame = () => {
  const [game, setGame] = useState({
    board: [],
    turn: 1,
    winner: null,
    width: 13,
    height: 9,
    lines: [],
    message: null
  });
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetchGame();
  }, []);

  const fetchGame = async () => {
    try {
      const response = await axios.get('http://localhost:5000/game');
      setGame(response.data);
    } catch (error) {
      console.error('Error fetching game state:', error);
    }
  };

  const handleClick = async (row, col) => {
    if (!selected) {
      if (game.board[row][col] !== ' ') {
        setSelected({ row, col });
      }
    } else {
      try {
        const response = await axios.post('http://localhost:5000/move', {
          from_row: selected.row,
          from_col: selected.col,
          to_row: row,
          to_col: col
        });
        if (response.data.error) {
          console.error(response.data.error);
        } else {
          setGame(response.data);
          setSelected(null);
        }
      } catch (error) {
        console.error('Error making move:', error);
      }
    }
  };

  const renderCell = (cell) => {
    switch (cell) {
      case 'B': return '•';
      case '1': return '1';
      case '2': return '2';
      case ' ': return '';
      default: return cell;
    }
  };

  const renderLines = () => {
    const cellSize = 30;
    return game.lines.map((line, index) => {
      const [start, end] = line;
      return (
        <line
          key={index}
          x1={(start[0] + 0.7) * cellSize}
          y1={(start[1] + 0.7) * cellSize}
          x2={(end[0] + 0.7) * cellSize}
          y2={(end[1] + 0.7) * cellSize}
          stroke="black"
          strokeWidth="3"
        />
      );
    });
  };

  return (
    <div className="game-container">
      <h1>Football Game</h1>
      <FieldSizeSelector onSizeChange={fetchGame} />
      {game.message && <div className="message">{game.message}</div>}
      <div className="board-container">
        <div
          className="board"
          style={{
            gridTemplateColumns: `repeat(${game.width}, 30px)`,
            gridTemplateRows: `repeat(${game.height}, 30px)`
          }}
        >
          {game.board.map((row, i) =>
            row.map((cell, j) => (
              <div
                key={`${i}-${j}`}
                onClick={() => handleClick(i, j)}
                className={`cell ${selected && selected.row === i && selected.col === j ? 'selected' : ''}`}
              >
                {renderCell(cell)}
              </div>
            ))
          )}
        </div>
        <svg
          className="lines-layer"
          width={game.width * 30}
          height={game.height * 30}
        >
        {renderLines()}
      </svg>
      <div className={`turn-display ${game.turn === 1 ? 'player1-turn' : 'player2-turn'}`}>
        Turn: Player {game.turn}
      </div>
    </div>
  </div>
  );
};



export default FootballGame;