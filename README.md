# AI Football Game

An interactive football strategy game with AI opponent, built using Python and React.

## Project Overview

AI Football Game is a turn-based strategy game simulating simplified football that was often played by me and my friends on paper. Players compete against a trained AI on a grid-based field. The project combines a Python backend with a React frontend.

### Key Features

- Custom game environment with implemented rules and mechanics
- Deep Q-Network (DQN) trained AI opponent
- Flask-based API for game interaction
- React frontend for user interface
- Player vs AI gameplay

## Components

1. Backend (Python):
   - `FootballGame.py`: Game environment and rules
   - `DQN_training.py`: DQN algorithm implementation and AI training
   - `live_game.py`: `AIPlayer` class for AI decision-making
   - `app.py`: Flask application serving as backend API

2. Frontend (React):
   - User interface for game interaction
   - Visual representation of the game board
   - Move input and game state display

## How It Works

1. Game initializes with specified board size
2. Players alternate turns, selecting start and end positions for the ball
3. Game validates moves, updates state, and checks for goals/game end
4. AI opponent uses Deep Q Networks for move selection
5. Flask API facilitates communication between frontend and game logic
6. React frontend provides user interface for gameplay

## Setup and Running

### Backend

1. Install required Python dependencies:
