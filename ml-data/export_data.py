import re
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import List

import orjson


@dataclass
class AdaptData:
    iterations: List["StateData"] = field(default_factory=list)
    coefficients: List[float] = field(default_factory=list)

    def explode_token(self, token: str, explode_qe=True) -> List[str]:
        # token has the form OVP/MVP(...)
        def OVP(*args):
            *ops, sign = args
            sign = "+" if sign == 1 else "-"
            tokens = [f"<ovp {sign}>"]

            for l in ops:
                tokens.extend(l)

            tokens.append("</ovp>")
            return tokens

        def MVP(*ops):
            tokens = [f"<mvp>"]

            for l in ops:
                tokens.extend(l)

            tokens.append("</mvp>")
            return tokens

        if explode_qe:

            def QE(*args):
                tokens = ["<qe>"]

                for arg in args:
                    tokens.append(str(arg))

                tokens.append("</qe>")
                return tokens

        else:

            def QE(*args):
                token = f"QE({', '.join(map(str, args))})"
                return [token]

        return eval(token)

    def to_tokens(self, explode_qe=True) -> List:
        tokens = []

        for data in self.iterations:
            tokens.extend(self.explode_token(data.main_operator, explode_qe=explode_qe))
            tokens.append("<tetris>")
            for token in data.tetris_operators:
                tokens.extend(self.explode_token(token, explode_qe=explode_qe))
            tokens.append("</tetris>")

        return tokens

    def token_coeffs(self, tokens: List[str]) -> List[float]:
        results = []

        i = 0
        mode = None
        for token in tokens:
            if token in {"<ovp +>", "<ovp ->"}:
                mode = "ovp"
            elif token == "<mvp>":
                mode = "mvp"

            if token.startswith("QE"):
                results.append(self.coefficients[int(i)])
                i += 0.5 if mode == "ovp" else 1
            else:
                results.append(0)

        return results


@dataclass
class StateData:
    main_operator: str = None
    tetris_operators: List[str] = field(default_factory=list)

    def __len__(self):
        return 1 + len(self.tetris_operators)


class State:
    Misc = 0
    NewIteration = 1
    AwaitingCeoType = 2
    ParsingOperator = 3
    Finalizing = 4
    Done = 5


class Patterns:
    Sep = re.compile(r"[\s,]+")
    NewIteration = re.compile(r"^\*\*\* ADAPT-VQE Iteration")

    AddingCeo = re.compile(r"Adding (OVP|MVP)-CEO\.")
    AddedOperators = re.compile(r"^Operator\(s\) added to ansatz: (\[.+\])")
    TetrisOperators = re.compile(r"^\d+ viable candidates: (\[.+\])")

    FinalEnergy = re.compile(r"^Final Energy")
    FinalCoefficients = re.compile(r"^Coefficients: (\[.+\])")
    FinalOperators = re.compile(r"^Final operators in the ansatz")
    Operator = re.compile(r"(OVP|MVP)\(.+\)")


def parse_log(text: str) -> AdaptData:
    adapt_data = AdaptData()
    state = State.Misc
    data = StateData()
    tetris = False

    # State transition function with some bookkeeping about
    # when to save our Adapt iteration data
    def transition_state(new_state):
        nonlocal state, data, tetris
        if new_state in {State.NewIteration, State.Finalizing, State.Done}:
            if data.main_operator is not None:
                adapt_data.iterations.append(data)

            data = StateData()
            tetris = False

        state = new_state

    # Loop over all lines
    for line in text.splitlines():
        line = line.strip()

        # Generic control flow
        if state != State.Finalizing:
            if match := Patterns.NewIteration.match(line):
                transition_state(State.NewIteration)
                continue
            elif match := Patterns.FinalEnergy.match(line):
                transition_state(State.Finalizing)
                continue

        # Specific states and transitions
        if state == State.NewIteration:
            transition_state(State.AwaitingCeoType)
            continue

        elif state == State.AwaitingCeoType:
            if match := Patterns.AddingCeo.match(line):
                transition_state(State.ParsingOperator)
                continue
            elif match := Patterns.TetrisOperators.match(line):
                tetris = True
                continue

        elif state == State.ParsingOperator:
            if match := Patterns.Operator.match(line):
                if tetris:
                    data.tetris_operators.append(match.group())
                else:
                    data.main_operator = match.group()

                transition_state(State.AwaitingCeoType)
                continue

        elif state == State.Finalizing:
            if match := Patterns.FinalCoefficients.match(line):
                s = match.group(1)[1:-1]
                adapt_data.coefficients.extend(map(float, Patterns.Sep.split(s)))
                transition_state(State.Done)
                continue

    assert state == State.Done
    # Check that all the ansatz chunks match the total number of coefficients we read
    # assert len(adapt_data.coefficients) == sum(len(x) for x in adapt_data.iterations)
    return adapt_data


def parse_log_file(logfile: Path) -> AdaptData:
    try:
        adapt_data = parse_log(logfile.read_text())
        return adapt_data
    except Exception as e:
        print(f"Error parsing {logfile}: {e}")


def export_files(directory: Path, processes=None):
    # Load all of the files in the log directory and process them in parallel
    logfiles = list(directory.glob("*.txt"))
    with Pool(processes) as pool:
        adapt_data = pool.map(parse_log_file, logfiles)

    # Export the tokens to another directory
    dataset_dir = Path("dataset")
    dataset_dir.mkdir(exist_ok=True)
    for logfile, data in zip(logfiles, adapt_data):
        if data:
            examplefile = dataset_dir / f"{logfile.stem}.json"
            tokens = data.to_tokens(explode_qe=False)
            coefficients = data.token_coeffs(tokens)

            d = {"tokens": tokens, "coefficients": coefficients}
            examplefile.write_bytes(orjson.dumps(d))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=Path, default=Path("logs"))
    parser.add_argument("--processes", "-p", type=int, default=None)
    args = parser.parse_args()
    export_files(args.dir, args.processes)
