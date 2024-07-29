from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from live_game import AIPlayer
from FootballGame import FootballGameEnv

app = Flask(__name__)
CORS(app)

game = None

app = Flask(__name__)
CORS(app)

game = None
ai_player = None

def create_game_state(width, height):
    global game, ai_player
    game = FootballGameEnv(height=height, width=width)
    obs = game.reset()


    input_shape = obs.shape
    n_actions = game.action_space.n
    ai_player = AIPlayer(input_shape, n_actions, "trained_agent2.pth")

    return {
        "board": game.game_state.board.tolist(),
        "turn": game.turn,
        "width": game.width,
        "height": game.height,
        "lines": list(game.game_state.lines),
        "winner": game.winner,
        "message": None
    }

game_state = create_game_state(13, 13)

@app.route("/game", methods=["GET"])
def get_game_state():
    global game
    return jsonify({
        "board": game.game_state.board.tolist(),
        "turn": game.turn,
        "width": game.width,
        "height": game.height,
        "lines": list(game.game_state.lines),
        "winner": game.winner
    })

@app.route("/set_field_size", methods=["POST"])
def set_field_size():
    data = request.json
    width = data.get("width", 13)
    height = data.get("height", 9)

    global game_state
    game_state = create_game_state(width, height)
    return jsonify(game_state)

@app.route("/move", methods=["POST"])
def make_move():
    global game, ai_player
    data = request.json
    from_pos = (data.get("from_col"), data.get("from_row"))
    to_pos = (data.get("to_col"), data.get("to_row"))

    action = game._get_action_from_positions(from_pos, to_pos)
    observation, reward, done, info = game.step(action)

    if done:
        winner = game.winner
        response = create_game_state(game.width, game.height)
        response["message"] = f"Player {winner} won! The game has been reset."
        return jsonify(response)


    while game.turn == 2:
        print("AI's turn")
        legal_actions = game.get_legal_actions()
        ai_action = ai_player.select_action(game._get_observation(), legal_actions)

        observation, reward, done, info = game.step(ai_action)
        print(f"Game turn after AI move: {game.turn}")
        print("reward:", reward)
        print("done:", done)
        print("winner:", game.winner)
        print(f"info: {info}")
        if done:
            winner = game.winner
            response = create_game_state(game.width, game.height)
            response["message"] = f"Player {winner} won! The game has been reset."
            return jsonify(response)

        print("Extra move granted:", game.extra_move_granted)
        if not game.extra_move_granted:
            break

    return jsonify({
        "board": game.game_state.board.tolist(),
        "turn": game.turn,
        "width": game.width,
        "height": game.height,
        "lines": list(game.game_state.lines),
        "winner": game.winner,
        "extraMove": game.extra_move_granted,
        "message": "Extra move granted!" if game.extra_move_granted else None
    })

if __name__ == "__main__":
    app.run(debug=True)