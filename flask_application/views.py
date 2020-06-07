from flask_application import app

import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import numpy as np

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, PROJECT_DIR+"/tindar-engine")

from tindar import Tindar, TindarGenerator


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html"), 200


@app.route('/api', methods=["GET"])
def api_home():
    return render_template("api_explanation.html"), 200


@app.route('/api/generate', methods=["GET"])
def generate_tindar_problem():
    args = request.args

    if not args:  # query string is empty
        return render_template("api_explanation.html"), 200

    param_count = 0
    for k, v in args.items():
        if k == "n":
            try:
                n = int(v)
            except Exception:
                return f"Did not specify n right: {v}", 400
            param_count += 1
        elif k == "connectedness":
            try:
                connectedness = float(v)
            except Exception:
                return f"Did not specify connectedness right: {v}"
            param_count += 1
        else:
            return f"parameter '{k}' in query string not allowed, only 'n' and 'connectedness'", 400
    if param_count != 2:
        return "did not specify n and connectedness", 400

    try:
        tindar_problem = TindarGenerator(n, connectedness)
    except Exception as e:
        return str(e), 400

    rv = {
        "n": n,
        "connectedness": connectedness,
        "love_matrix": tindar_problem.love_matrix.tolist(),
        "p": tindar_problem.p
    }

    return jsonify(rv), 200


@app.route("/api/solve", methods=["GET", "POST"])
def solve_tindar_problem():
    if request.method == "GET":
        # TODO: EXPLAIN HOW TO INTERACT
        return render_template("api_explanation.html")

    elif request.method == "POST":
        if request.is_json:
            req = request.get_json()

            # Parse parameters from JSON
            param_count = 0
            for k, v in req.items():
                if k == "love_matrix":
                    try:
                        love_matrix_np = np.array(v)
                        param_count += 1
                    except Exception as e:
                        return f"Cannot convert love_matrix to np_array: {love_matrix}. {e}", 400

                elif k == "solver":
                    solver = v
                    if solver not in ["pulp", "heuristic"]:
                        return f"Solver {solver} not allowed. Choose 'pulp', 'heuristic'", 400
                    else:
                        param_count += 1

                else:
                    return f"Parameter {k} not allowed", 400

            if param_count != 2:
                return "Did not specify all parameters: love_matrix, solver", 400

            # Solve problem
            try:
                tindar = Tindar(love_matrix_np)
            except Exception as e:
                return f"Could not initialize Tindar object: {e}"

            if solver == "pulp":
                tindar.create_problem()

            tindar.solve_problem(kind=solver)
            obj = tindar.solution_obj(kind=solver, verbose=False)
            sol = tindar.solution_vars(kind=solver, verbose=False).astype(int).tolist()

            return jsonify({
                "people_paired": obj,
                "solution": sol
            })

        else:
            return "/api/solve only accepts JSON POST's", 400

    else:
        return "/api/solve only accepts GET and POST requests", 400


if __name__ == '__main__':
    app.run(debug=False, host='localhost', port="8080", threaded=True)
