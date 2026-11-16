import pytest
import numpy as np

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.piece import Piece
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


# ---------------------------------------------------------------- clean_board
def test_clean_board_shape_and_types():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Piece)


def test_clean_board_all_empty_with_positions():
    b = clean_board()
    for i in range(8):
        for j in range(8):
            assert b[i, j].team is None
            assert b[i, j].get_position() == (i, j)


def test_clean_board_returns_independent_arrays():
    a = clean_board()
    b = clean_board()
    a[0, 0] = Knight((0, 0), "white")
    assert b[0, 0].team is None


# --------------------------------------------------------------------- Knight
def test_knight_name_value_symbol():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_moves_from_start_position_white():
    board = Board()
    knight = board.board[0, 1]
    moves = knight.get_moves([board])
    ends = sorted(m[1] for m in moves)
    assert ends == [(2, 0), (2, 2)]
    for m in moves:
        assert m[0] == (0, 1)
        assert m[2] == "base"


def test_knight_moves_from_start_position_black():
    board = Board()
    knight = board.board[7, 6]
    moves = knight.get_moves([board])
    ends = sorted(m[1] for m in moves)
    assert ends == [(5, 5), (5, 7)]


def test_knight_center_has_eight_moves():
    board = empty_board()
    board.board[4, 4] = Knight((4, 4), "white")
    moves = board.board[4, 4].get_moves([board])
    ends = set(m[1] for m in moves)
    expected = {(2, 3), (2, 5), (3, 2), (5, 2), (6, 3), (6, 5), (5, 6), (3, 6)}
    assert len(moves) == 8
    assert ends == expected


def test_knight_corner_and_capture_rules():
    board = empty_board()
    board.board[0, 0] = Knight((0, 0), "white")
    board.board[2, 1] = Pawn((2, 1), "black")   # capturable
    board.board[1, 2] = Pawn((1, 2), "white")   # blocked by own piece
    moves = board.board[0, 0].get_moves([board])
    ends = set(m[1] for m in moves)
    assert ends == {(2, 1)}


def test_knight_copy_is_independent():
    k = Knight((3, 3), "black")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "black"
    assert c.get_position() == (3, 3)
    assert c.moved is True
    c.set_position((4, 4))
    assert k.get_position() == (3, 3)


def test_knight_invalid_team_raises():
    with pytest.raises(ValueError):
        k = Knight((0, 0), "green")
        k.get_moves([empty_board()])


# ---------------------------------------------------------------------- Board
def test_board_default_setup():
    b = Board()
    assert b.turn == "white"
    assert b.board.shape == (8, 8)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[0, 3], Queen)
    assert isinstance(b.board[7, 0], Rook)
    assert isinstance(b.board[0, 1], Knight)
    for j in range(8):
        assert isinstance(b.board[1, j], Pawn)
        assert b.board[1, j].team == "white"
        assert isinstance(b.board[6, j], Pawn)
        assert b.board[6, j].team == "black"
    for i in range(2, 6):
        for j in range(8):
            assert b.board[i, j].team is None


def test_board_update_base_move():
    b = Board()
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(b.board[3, 4], Pawn)
    assert b.board[3, 4].get_position() == (3, 4)
    assert b.board[3, 4].moved is True
    assert b.board[1, 4].team is None
    assert b.turn == "black"
    b.update_board(((6, 4), (4, 4), "base"), "black")
    assert b.turn == "white"


def test_board_update_none_keyword_acts_as_base():
    b = Board()
    b.update_board(((0, 1), (2, 2), None), "white")
    assert isinstance(b.board[2, 2], Knight)
    assert b.board[0, 1].team is None


def test_board_update_enpasse_removes_captured_pawn():
    b = empty_board()
    b.board[4, 4] = Pawn((4, 4), "white")
    b.board[4, 3] = Pawn((4, 3), "black")
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert b.board[4, 3].team is None
    assert b.board[4, 4].team is None


def test_board_update_castling_kingside_moves_rook():
    b = empty_board()
    b.board[0, 4] = King((0, 4), "white")
    b.board[0, 7] = Rook((0, 7), "white")
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert b.board[0, 4].team is None
    assert b.board[0, 7].team is None


def test_board_update_castling_queenside_moves_rook():
    b = empty_board()
    b.board[7, 4] = King((7, 4), "black")
    b.board[7, 0] = Rook((7, 0), "black")
    b.update_board(((7, 4), (7, 2), "castling"), "black")
    assert isinstance(b.board[7, 2], King)
    assert isinstance(b.board[7, 3], Rook)
    assert b.board[7, 0].team is None


def test_board_update_evolve_creates_queen():
    b = empty_board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert b.board[6, 0].team is None


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert b.equal(c)
    assert c.turn == b.turn
    assert c.board[0, 1] is not b.board[0, 1]
    c.update_board(((1, 0), (3, 0), "base"), "white")
    assert not b.equal(c)
    # original unaffected
    assert isinstance(b.board[1, 0], Pawn)
    assert b.turn == "white"


def test_board_str_contains_labels():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    for n in "12345678":
        assert n in s


# ----------------------------------------------------------------- GameEngine
def test_engine_initial_state():
    e = GameEngine(Board())
    assert e.move_counter == 0
    assert len(e.game_states) == 1
    assert e.get_turn() == ("white", "human")
    assert e.white_check is False
    assert e.black_check is False


def test_engine_get_turn_alternates_and_players():
    e = GameEngine(Board(), "computer", "human")
    assert e.get_turn() == ("white", "computer")
    e.move_counter = 1
    assert e.get_turn() == ("black", "human")


def test_engine_num_alph_conversion():
    result = GameEngine.num_alph({1: ((1, 4), (3, 4), "base"),
                                  2: ((0, 0), (7, 7), "base")})
    assert result == {1: ["E2", "E4"], 2: ["A1", "H8"]}


def test_engine_get_moves_and_all_moves_default():
    e = GameEngine(Board())
    pawn_moves = e.get_moves((1, 4))
    ends = sorted(m[1] for m in pawn_moves)
    assert ends == [(2, 4), (3, 4)]
    assert len(e.all_moves()) == 20


def test_engine_computer_move_updates_state_and_undo():
    e = GameEngine(Board(), "computer", "computer")
    e.choose_move(((1, 4), (3, 4), "base"))
    assert e.move_counter == 1
    assert len(e.game_states) == 2
    assert isinstance(e.Board.board[3, 4], Pawn)
    assert e.get_turn()[0] == "black"
    e.undo_move()
    assert len(e.game_states) == 1
    assert e.Board.board[3, 4].team is None
    assert isinstance(e.Board.board[1, 4], Pawn)


def test_engine_choose_move_invalid_modes_raise():
    e = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        e.choose_move(None)
    e2 = GameEngine(Board(), "human", "human")
    with pytest.raises(ValueError):
        e2.choose_move(((1, 4), (3, 4), "base"))
    e3 = GameEngine(Board(), "robot", "robot")
    with pytest.raises(ValueError):
        e3.choose_move(((1, 4), (3, 4), "base"))


def test_engine_is_check_detects_rook_check():
    b = empty_board()
    b.board[0, 4] = King((0, 4), "white")
    b.board[7, 4] = Rook((7, 4), "black")
    b.board[7, 0] = King((7, 0), "black")
    e = GameEngine(b)
    e.is_check()
    assert e.white_check is True
    assert e.black_check is False


def test_engine_is_check_false_on_default_board():
    e = GameEngine(Board())
    e.is_check()
    assert not e.white_check
    assert not e.black_check


def test_engine_stalemate_checks():
    e = GameEngine(Board())
    # normal position: no exit
    e.is_stalemate_no_move("white")
    e.is_stalemate_3states("white")
    assert e.compare_states(e.game_states[0], e.Board)
    # nine identical states trigger a stalemate exit
    e.game_states = [e.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        e.is_stalemate_3states("white")
