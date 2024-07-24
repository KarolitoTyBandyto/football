from flask import Flask, request, jsonify
from flask_cors import CORS
from FootballGame import FootballGame
import numpy as np

app = Flask(__name__)
CORS(app)

game = None

def create_game_state(width, height):
    global game
    game = FootballGame(height, width)
    return {
        "board": game.board.tolist(),
        "turn": game.turn,
        "width": game.pitch_width,
        "height": game.pitch_height,
        "lines": list(game.lines),  # Convert set to list
        "winner": game.winner,
        "message": None
    }

game_state = create_game_state(13, 9)

@app.route("/game", methods=["GET"])
def get_game_state():
    global game
    return jsonify({
        "board": game.board.tolist(),
        "turn": game.turn,
        "width": game.pitch_width,
        "height": game.pitch_height,
        "lines": list(game.lines),  # Convert set to list
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
    global game
    data = request.json
    from_pos = (data.get("from_col"), data.get("from_row"))
    to_pos = (data.get("to_col"), data.get("to_row"))

    if game.make_move(from_pos, to_pos):
        if game.winner is not None:
            winner = game.winner
            print(winner)
            response = create_game_state(game.pitch_width, game.pitch_height)
            response["message"] = f"Player {winner} won! The game has been reset."
            return jsonify(response)
        elif game.extra_move_granted:
            response = {
                "board": game.board.tolist(),
                "turn": game.turn,
                "width": game.pitch_width,
                "height": game.pitch_height,
                "lines": list(game.lines),
                "winner": game.winner,
                "extraMove": True,
                "message": "Extra move granted! You hit a post or connected to a previous line."
            }
            return jsonify(response)
        else:
            game.turn = 2 if game.turn == 1 else 1

        return jsonify({
            "board": game.board.tolist(),
            "turn": game.turn,
            "width": game.pitch_width,
            "height": game.pitch_height,
            "lines": list(game.lines),
            "winner": game.winner,
            "extraMove": False
        })
    else:
        return jsonify({"error": "Invalid move."}), 400

if __name__ == "__main__":
    app.run(debug=True)