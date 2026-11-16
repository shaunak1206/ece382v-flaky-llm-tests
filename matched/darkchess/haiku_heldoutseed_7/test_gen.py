import sys

import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen


def _single_state(arr):
    """Wrap a numpy board in a Board object, as Knight.get_moves expects game_states[-1].board."""
    return [Board(starting_board=arr)]


# ---------------------------------------------------------------- Knight

def test_knight_name_and_value():
    knight = Knight((0, 0), "white")
    assert knight.get_name() == "knight"
    assert knight.get_value() == 3


def test_knight_symbol_for_each_team():
    assert Knight((0, 1), "white").get_symbol() == "wk "
    assert Knight((0, 1), "black").get_symbol() == "bk "


def test_knight_symbol_invalid_team_raises():
    with pytest.raises(ValueError):
        Knight((0, 1), "red").get_symbol()


def test_knight_moves_centre_on_empty_board_has_eight_moves():
    arr = clean_board()
    knight = Knight((3, 3), "white")
    arr[3, 3] = knight
    moves = knight.get_moves(_single_state(arr))
    ends = sorted(m[1] for m in moves)
    expected = sorted([(1, 2), (1, 4), (2, 1), (2, 5),
                       (4, 1), (4, 5), (5, 2), (5, 4)])
    assert ends == expected
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_moves_corner_has_two_moves():
    arr = clean_board()
    knight = Knight((0, 0), "white")
    arr[0, 0] = knight
    ends = sorted(m[1] for m in knight.get_moves(_single_state(arr)))
    assert ends == [(1, 2), (2, 1)]


def test_knight_moves_excludes_own_team_piece():
    arr = clean_board()
    knight = Knight((3, 3), "white")
    arr[3, 3] = knight
    arr[5, 4] = Knight((5, 4), "white")  # blocked by own piece
    ends = [m[1] for m in knight.get_moves(_single_state(arr))]
    assert len(ends) == 7
    assert (5, 4) not in ends


def test_knight_copy_is_independent_object():
    knight = Knight((2, 2), "black")
    knight.moved = True
    clone = knight.copy()
    assert clone is not knight
    assert isinstance(clone, Knight)
    assert clone.team == "black"
    assert clone.moved is True
    assert tuple(clone.get_position()) == (2, 2)


# ---------------------------------------------------------------- utils.clean_board

def test_clean_board_shape_and_dummy_pieces():
    arr = clean_board()
    assert arr.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(arr[i, j], Dummy)


def test_clean_board_positions_match_indices():
    arr = clean_board()
    for i in range(8):
        for j in range(8):
            assert tuple(arr[i, j].get_position()) == (i, j)


# ---------------------------------------------------------------- Board

def test_default_board_setup_and_turn():
    board = Board()
    assert board.turn == "white"
    assert isinstance(board.board[1, 0], Pawn)
    assert board.board[1, 0].team == "white"
    assert isinstance(board.board[6, 0], Pawn)
    assert board.board[6, 0].team == "black"
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[7, 3], Queen)
    assert isinstance(board.board[3, 3], Dummy)


def test_update_board_base_move_moves_piece_and_switches_turn():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(board.board[3, 4], Pawn)
    assert isinstance(board.board[1, 4], Dummy)
    assert tuple(board.board[3, 4].get_position()) == (3, 4)
    assert board.board[3, 4].moved is True
    assert board.turn == "black"


def test_update_board_none_keyword_defaults_to_base():
    board = Board()
    board.update_board(((1, 4), (3, 4), None), "white")
    assert isinstance(board.board[3, 4], Pawn)
    assert isinstance(board.board[1, 4], Dummy)


def test_update_board_castling_kingside_moves_rook():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[0, 7] = Rook((0, 7), "white")
    board = Board(starting_board=arr)
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)
    assert tuple(board.board[0, 5].get_position()) == (0, 5)


def test_update_board_enpasse_removes_captured_pawn():
    arr = clean_board()
    arr[4, 4] = Pawn((4, 4), "white")
    arr[4, 5] = Pawn((4, 5), "black")
    board = Board(starting_board=arr)
    board.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert isinstance(board.board[5, 5], Pawn)
    assert board.board[5, 5].team == "white"
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)


def test_update_board_evolve_promotes_to_queen():
    arr = clean_board()
    arr[6, 0] = Pawn((6, 0), "white")
    board = Board(starting_board=arr)
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert isinstance(board.board[6, 0], Dummy)


def test_update_board_invalid_turn_raises_value_error():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), "base"), "white")


def test_board_copy_is_deep():
    original = Board()
    clone = original.copy()
    assert clone is not original
    assert clone.board[1, 0] is not original.board[1, 0]
    clone.board[1, 0].moved = True
    assert original.board[1, 0].moved is False
    assert original.equal(clone)


def test_board_equal_detects_difference():
    original = Board()
    changed = original.copy()
    changed.update_board(((1, 0), (2, 0), "base"), "white")
    assert not original.equal(changed)


# ---------------------------------------------------------------- GameEngine

def test_get_turn_alternates_and_reports_players():
    engine = GameEngine(Board(), player1="computer", player2="human")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_num_alph_converts_to_chess_notation():
    result = GameEngine.num_alph({1: ((1, 4), (3, 4), "base")})
    assert result == {1: ["E2", "E4"]}


def test_get_moves_for_knight_on_starting_board():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    ends = sorted(m[1] for m in moves)
    assert ends == [(2, 0), (2, 2)]


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert isinstance(engine.Board.board[1, 4], Pawn)
    assert isinstance(engine.Board.board[3, 4], Dummy)


def test_is_check_detects_knight_attacking_king():
    arr = clean_board()
    arr[0, 0] = King((0, 0), "white")
    arr[5, 4] = King((5, 4), "black")
    arr[3, 3] = Knight((3, 3), "white")  # attacks (5, 4)
    engine = GameEngine(Board(starting_board=arr))
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_is_stalemate_no_move_exits_when_team_cannot_move():
    arr = clean_board()
    arr[7, 7] = King((7, 7), "black")  # white has no pieces at all
    engine = GameEngine(Board(starting_board=arr))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_choose_move_computer_applies_move():
    engine = GameEngine(Board(), player1="computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert engine.Board.turn == "black"
