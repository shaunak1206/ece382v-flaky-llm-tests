import sys

import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def _state(arr):
    """Wrap a raw numpy board in a Board so piece move generators can read it."""
    return [Board(arr)]


# ---------------------------------------------------------------- Board

def test_default_board_pieces():
    board = Board()
    white = [p for p in board.board.flat if p.team == "white"]
    black = [p for p in board.board.flat if p.team == "black"]
    assert len(white) == 16
    assert len(black) == 16
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 0], Rook)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[1, 0], Pawn)
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[7, 3].team == "black"


def test_initial_turn_is_white():
    assert Board().turn == "white"


def test_clean_board_all_empty_cells():
    arr = clean_board()
    assert arr.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(arr[i, j], Dummy)
            assert arr[i, j].team is None


def test_update_board_base_move():
    arr = clean_board()
    arr[0, 1] = Knight((0, 1), "white")
    board = Board(arr)
    board.update_board(((0, 1), (2, 2), "base"), "white")
    moved = board.board[2, 2]
    assert isinstance(moved, Knight)
    assert moved.get_position() == (2, 2)
    assert moved.moved is True
    assert board.board[0, 1].team is None
    assert board.turn == "black"


def test_update_board_invalid_turn_raises():
    board = Board(clean_board())
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((0, 0), (1, 1), "base"), "white")


def test_update_board_castling_kingside():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[0, 7] = Rook((0, 7), "white")
    board = Board(arr)
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert board.board[0, 5].moved is True
    assert board.board[0, 4].team is None
    assert board.board[0, 7].team is None


def test_update_board_enpassant_removes_pawn():
    arr = clean_board()
    arr[4, 3] = Pawn((4, 3), "white")
    arr[4, 4] = Pawn((4, 4), "black")
    board = Board(arr)
    board.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert board.board[5, 4].team == "white"
    assert isinstance(board.board[5, 4], Pawn)
    assert board.board[4, 4].team is None
    assert board.board[4, 3].team is None


def test_update_board_evolve_creates_queen():
    arr = clean_board()
    arr[6, 0] = Pawn((6, 0), "white")
    board = Board(arr)
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert board.board[6, 0].team is None


def test_board_copy_is_independent():
    original = Board()
    original.turn = "black"
    dup = original.copy()
    assert dup is not original
    assert dup.turn == "black"
    assert dup.board[1, 0] is not original.board[1, 0]
    dup.board[1, 0] = Dummy((1, 0))
    dup.turn = "white"
    assert original.board[1, 0].team == "white"
    assert original.turn == "black"


def test_board_equal_detects_difference():
    b1 = Board()
    b2 = b1.copy()
    assert b1.equal(b2)
    b2.board[1, 0] = Knight((1, 0), "white")
    assert not b1.equal(b2)


def test_board_str_layout():
    lines = str(Board()).splitlines()
    assert len(lines) == 10
    assert lines[0].startswith("  A  B")
    assert lines[-1].startswith("  A  B")
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")


# ---------------------------------------------------------------- GameEngine

def test_get_turn_alternates():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    engine2 = GameEngine(Board(), player1="computer", player2="human")
    assert engine2.get_turn() == ("white", "computer")


def test_get_moves_knight_from_start():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    ends = {m[1] for m in moves}
    assert ends == {(2, 0), (2, 2)}
    for m in moves:
        assert m[0] == (0, 1)
        assert m[2] == "base"


def test_num_alph_converts_moves():
    result = GameEngine.num_alph({1: ((0, 1), (2, 0), "base")})
    assert result == {1: ["B1", "A3"]}


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((1, 0), (3, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.board[1, 0].team == "white"
    assert engine.Board.board[3, 0].team is None


def test_is_check_detects_knight_attack():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[2, 3] = Knight((2, 3), "black")  # knight on (2,3) attacks (0,4)
    engine = GameEngine(Board(arr))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_stalemate_no_move_exits_when_no_moves():
    engine = GameEngine(Board(clean_board()))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_stalemate_3states_exits_on_repetition():
    engine = GameEngine(Board(clean_board()))
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_choose_move_computer_none_raises():
    engine = GameEngine(Board(), player1="computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)


def test_computer_move_applies_and_advances_turn():
    engine = GameEngine(Board(), player1="computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.Board.board[3, 4].team == "white"
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2


# ---------------------------------------------------------------- Knight

def test_knight_moves_center_empty_board():
    arr = clean_board()
    knight = Knight((3, 3), "white")
    arr[3, 3] = knight
    ends = {m[1] for m in knight.get_moves(_state(arr))}
    assert ends == {(1, 2), (1, 4), (2, 1), (2, 5),
                    (4, 1), (4, 5), (5, 2), (5, 4)}


def test_knight_does_not_capture_own_team():
    arr = clean_board()
    knight = Knight((3, 3), "white")
    arr[3, 3] = knight
    arr[5, 4] = Knight((5, 4), "white")   # own piece, must be excluded
    arr[1, 2] = Knight((1, 2), "black")   # enemy piece, capture allowed
    moves = knight.get_moves(_state(arr))
    ends = {m[1] for m in moves}
    assert len(moves) == 7
    assert (5, 4) not in ends
    assert (1, 2) in ends


def test_knight_symbol_and_invalid_team():
    assert Knight((0, 0), "white").get_symbol() == "wk "
    assert Knight((0, 0), "black").get_symbol() == "bk "
    bad = Knight((0, 0), "red")
    with pytest.raises(ValueError):
        bad.get_symbol()
    with pytest.raises(ValueError):
        bad.get_moves(_state(clean_board()))


def test_knight_copy():
    knight = Knight((3, 3), "black")
    knight.moved = True
    dup = knight.copy()
    assert isinstance(dup, Knight)
    assert dup is not knight
    assert dup.get_position() == (3, 3)
    assert dup.team == "black"
    assert dup.moved is True
