import React from 'react';

const GameControls = ({ gameMode, playerNumber, onGameModeToggle, onPlayerNumberToggle }) => {
  return (
    <div className="game-controls">
      <button onClick={onGameModeToggle}>
        {gameMode === 'ai' ? 'Switch to PvP' : 'Switch to AI'}
      </button>
      {gameMode === 'ai' && (
        <button onClick={onPlayerNumberToggle}>
          Play as Player {playerNumber === 1 ? 2 : 1}
        </button>
      )}
    </div>
  );
};

export default GameControls;