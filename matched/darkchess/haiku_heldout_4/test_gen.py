import sys

import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.utils import clean_board


def empty_board():
    """Board with no pieces (only Dummy placeholders)."""
    return Board(clean_board())


# ---------------------------------------------------------------- Board

def test_board_default_turn_is_white():
    board = Board()
    assert board.turn == "white"


def test_default_board_pawns_and_pieces_placement():
    board = Board()
    for j in range(8):
        assert isinstance(board.board[1, j], Pawn)
        assert board.board[1, j].team == "white"
        assert isinstance(board.board[6, j], Pawn)
        assert board.board[6, j].team == "black"
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[0, 3], Queen)
    assert board.board[7, 4].team == "black"
    assert isinstance(board.board[7, 4], King)


def test_clean_board_shape_and_dummy_pieces():
    grid = clean_board()
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(grid[i, j], Dummy)
            assert grid[i, j].team is None


def test_board_accepts_custom_starting_board():
    grid = clean_board()
    grid[4, 4] = Knight((4, 4), "white")
    board = Board(grid)
    assert board.board is grid
    assert isinstance(board.board[4, 4], Knight)


def test_update_board_base_move_moves_piece():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(board.board[3, 4], Pawn)
    assert board.board[3, 4].team == "white"
    assert board.board[3, 4].get_position() == (3, 4)
    assert board.board[3, 4].moved is True
    assert isinstance(board.board[1, 4], Dummy)


def test_update_board_switches_turn():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    assert board.turn == "black"
    board.update_board(((6, 4), (4, 4), "base"), "black")
    assert board.turn == "white"


def test_update_board_invalid_turn_raises():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 4), (3, 4), "base"), "white")


def test_board_copy_is_equal_and_independent():
    board = Board()
    clone = board.copy()
    assert board.equal(clone)
    assert clone.turn == board.turn
    clone.update_board(((1, 4), (3, 4), "base"), "white")
    # Original must be untouched
    assert isinstance(board.board[1, 4], Pawn)
    assert not board.equal(clone)


def test_board_equal_detects_difference():
    a = Board()
    b = Board()
    assert a.equal(b)
    b.board[1, 0] = Dummy((1, 0), None)
    assert not a.equal(b)


def test_board_view_board_prints_labels(capsys):
    board = Board()
    board.view_board()
    out = capsys.readouterr().out
    assert "A  B  C  D  E  F  G  H" in out


# ---------------------------------------------------------------- Knight

def test_knight_name_and_value():
    knight = Knight((0, 1), "white")
    assert knight.get_name() == "knight"
    assert knight.get_value() == 3


def test_knight_symbol_by_team():
    assert Knight((0, 1), "white").get_symbol() == "wk "
    assert Knight((7, 1), "black").get_symbol() == "bk "


def test_knight_symbol_invalid_team_raises():
    with pytest.raises(ValueError):
        Knight((0, 1), "green").get_symbol()


def test_knight_moves_from_center_on_empty_board():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "white")
    moves = board.board[4, 4].get_moves([board])
    ends = {m[1] for m in moves}
    assert len(moves) == 8
    assert ends == {(2, 3), (2, 5), (3, 2), (5, 2),
                    (6, 3), (6, 5), (5, 6), (3, 6)}
    for start, _, kind in moves:
        assert start == (4, 4)
        assert kind == "base"


def test_knight_moves_from_corner_on_empty_board():
    board = empty_board()
    board.board[0, 0] = Knight((0, 0), "white")
    ends = {m[1] for m in board.board[0, 0].get_moves([board])}
    assert ends == {(2, 1), (1, 2)}


def test_knight_does_not_capture_own_piece():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "white")
    board.board[2, 3] = Knight((2, 3), "white")
    ends = {m[1] for m in board.board[4, 4].get_moves([board])}
    assert (2, 3) not in ends
    assert len(ends) == 7


def test_knight_can_capture_enemy_piece():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "black")
    board.board[2, 3] = Knight((2, 3), "white")
    ends = {m[1] for m in board.board[4, 4].get_moves([board])}
    assert (2, 3) in ends


def test_knight_get_moves_invalid_team_raises():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "green")
    with pytest.raises(ValueError):
        board.board[4, 4].get_moves([board])


def test_knight_copy_preserves_state_and_is_independent():
    original = Knight((3, 3), "black")
    original.moved = True
    clone = original.copy()
    assert isinstance(clone, Knight)
    assert clone is not original
    assert clone.team == "black"
    assert clone.get_position() == (3, 3)
    assert clone.moved is True
    clone.moved = False
    assert original.moved is True


# ---------------------------------------------------------------- GameEngine

def test_game_engine_initial_state():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False


def test_game_engine_get_turn_alternates():
    engine = GameEngine(Board(), player1="computer", player2="human")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_game_engine_num_alph_converts_moves():
    move_dict = {1: ((1, 4), (3, 4), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["E2", "E4"]}


def test_game_engine_get_moves_for_starting_knight():
    engine = GameEngine(Board())
    ends = {m[1] for m in engine.get_moves((0, 1))}
    # (1, 3) is occupied by a white pawn, so only two moves remain
    assert ends == {(2, 0), (2, 2)}


def test_game_engine_undo_move_restores_previous_board():
    engine = GameEngine(Board())
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert isinstance(engine.Board.board[1, 4], Pawn)
    assert isinstance(engine.Board.board[3, 4], Dummy)


def test_game_engine_is_check_detects_black_in_check():
    board = empty_board()
    board.board[4, 4] = King((4, 4), "black")
    board.board[2, 3] = Knight((2, 3), "white")  # attacks (4, 4)
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_game_engine_stalemate_when_no_moves_exits():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "black")  # white has no pieces at all
    engine = GameEngine(board)
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")
