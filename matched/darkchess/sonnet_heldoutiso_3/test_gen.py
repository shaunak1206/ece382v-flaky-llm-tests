from types import SimpleNamespace

import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.piece import Piece
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def _states_for(board_array):
    return [SimpleNamespace(board=board_array)]


def _end_positions(moves):
    return {m[1] for m in moves}


# ---------------------------- clean_board ----------------------------

def test_clean_board_shape_and_contents():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Piece)
            assert b[i, j].team is None


def test_clean_board_returns_independent_arrays():
    a = clean_board()
    b = clean_board()
    a[0, 0] = Knight((0, 0), "white")
    assert b[0, 0].team is None
    assert a is not b


# ------------------------------ Knight ------------------------------

def test_knight_basic_attributes():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "
    assert w.get_position() == (0, 1)


def test_knight_center_has_eight_moves_on_empty_board():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    moves = knight.get_moves(_states_for(board))
    assert len(moves) == 8
    assert _end_positions(moves) == {
        (1, 2), (1, 4), (2, 1), (4, 1), (5, 2), (5, 4), (2, 5), (4, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves():
    board = clean_board()
    knight = Knight((0, 0), "black")
    board[0, 0] = knight
    moves = knight.get_moves(_states_for(board))
    assert _end_positions(moves) == {(2, 1), (1, 2)}


def test_knight_cannot_capture_friendly_but_can_capture_enemy():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    board[5, 4] = Pawn((5, 4), "white")
    board[5, 2] = Pawn((5, 2), "black")
    ends = _end_positions(knight.get_moves(_states_for(board)))
    assert (5, 4) not in ends
    assert (5, 2) in ends
    assert len(ends) == 7


def test_knight_black_moves_blocked_by_black_pieces():
    board = clean_board()
    knight = Knight((4, 4), "black")
    board[4, 4] = knight
    board[6, 5] = Pawn((6, 5), "black")
    board[6, 3] = Pawn((6, 3), "white")
    ends = _end_positions(knight.get_moves(_states_for(board)))
    assert (6, 5) not in ends
    assert (6, 3) in ends


def test_knight_copy_is_independent():
    k = Knight((0, 1), "white")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "white"
    assert c.get_position() == (0, 1)
    assert c.moved is True
    c.set_position((2, 2))
    assert k.get_position() == (0, 1)


# ------------------------------- Board -------------------------------

def test_board_default_setup():
    b = Board()
    assert b.turn == "white"
    count = sum(1 for i in range(8) for j in range(8)
                if b.board[i, j].team is not None)
    assert count == 32
    assert isinstance(b.board[0, 1], Knight)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[0, 3], Queen)
    assert isinstance(b.board[7, 0], Rook)
    assert all(b.board[1, j].team == "white" for j in range(8))
    assert all(b.board[6, j].team == "black" for j in range(8))
    for j in range(8):
        assert b.board[3, j].team is None


def test_board_custom_starting_board():
    arr = clean_board()
    arr[0, 0] = King((0, 0), "white")
    b = Board(starting_board=arr)
    assert b.board is arr
    assert isinstance(b.board[0, 0], King)
    assert b.turn == "white"


def test_board_str_contains_labels_and_symbols():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    assert "wk " in s
    assert "bk " in s
    assert s.count("\n") == 10


def test_update_board_base_move():
    b = Board()
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(b.board[3, 4], Pawn)
    assert b.board[3, 4].get_position() == (3, 4)
    assert b.board[3, 4].moved is True
    assert b.board[1, 4].team is None
    assert b.turn == "black"
    b.update_board(((6, 4), (4, 4), None), "black")
    assert isinstance(b.board[4, 4], Pawn)
    assert b.turn == "white"


def test_update_board_enpassant_removes_captured_pawn():
    arr = clean_board()
    arr[4, 4] = Pawn((4, 4), "white")
    arr[4, 3] = Pawn((4, 3), "black")
    b = Board(starting_board=arr)
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert b.board[4, 3].team is None
    assert b.board[4, 4].team is None


def test_update_board_castling_kingside():
    b = Board()
    b.board[0, 5] = clean_board()[0, 5]
    b.board[0, 6] = clean_board()[0, 6]
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 7].team is None
    assert b.board[0, 4].team is None
    assert b.board[0, 5].get_position() == (0, 5)


def test_update_board_evolve_creates_queen():
    arr = clean_board()
    arr[6, 0] = Pawn((6, 0), "white")
    b = Board(starting_board=arr)
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert b.board[6, 0].team is None


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.board is not b.board
    assert b.equal(c)
    assert c.turn == b.turn
    c.update_board(((1, 0), (3, 0), "base"), "white")
    assert not b.equal(c)
    assert b.board[1, 0].team == "white"
    assert b.turn == "white"


# ---------------------------- GameEngine ----------------------------

def test_engine_init_defaults_and_turns():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    assert _end_positions(moves) == {(2, 0), (2, 2)}


def test_num_alph_conversion():
    result = GameEngine.num_alph({
        1: ((0, 0), (2, 0), "base"),
        2: ((0, 6), (2, 7), "base"),
    })
    assert result == {1: ["A1", "A3"], 2: ["G1", "H3"]}


def test_choose_move_computer_mode(capsys):
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert len(engine.game_states) == 2
    assert engine.get_turn()[0] == "black"


def test_choose_move_invalid_modes(capsys):
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    human = GameEngine(Board())
    with pytest.raises(ValueError):
        human.choose_move(((1, 4), (3, 4), "base"))
    bad = GameEngine(Board(), "alien", "human")
    with pytest.raises(ValueError):
        bad.choose_move(((1, 4), (3, 4), "base"))


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert not engine.Board.equal(original)
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)
    assert engine.Board.turn == "white"


def test_is_check_detects_knight_check():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[7, 4] = King((7, 4), "black")
    arr[2, 3] = Knight((2, 3), "black")
    engine = GameEngine(Board(starting_board=arr))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_is_check_false_at_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False


def test_stalemate_3states_no_exit_for_short_history():
    engine = GameEngine(Board())
    assert engine.is_stalemate_3states("white") is None
    assert engine.compare_states(engine.game_states[0], Board())


def test_stalemate_3states_exits_on_repeated_states(capsys):
    engine = GameEngine(Board())
    engine.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")
