import numpy as np
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


def empty_board():
    return Board(clean_board())


def end_squares(moves):
    return {m[1] for m in moves}


# ---------------------------------------------------------------- clean_board

def test_clean_board_is_8x8_of_dummies():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None
            assert b[i, j].get_position() == (i, j)


def test_default_board_back_rank_symbols():
    b = Board()
    white_row = [b.board[0, j].symbol for j in range(8)]
    black_row = [b.board[7, j].symbol for j in range(8)]
    assert white_row == ['wr ', 'wk ', 'wb ', 'wq ', 'wK ', 'wb ', 'wk ', 'wr ']
    assert black_row == ['br ', 'bk ', 'bb ', 'bq ', 'bK ', 'bb ', 'bk ', 'br ']


def test_default_board_pawns_and_empty_middle():
    b = Board()
    for j in range(8):
        assert isinstance(b.board[1, j], Pawn)
        assert b.board[1, j].team == "white"
        assert isinstance(b.board[6, j], Pawn)
        assert b.board[6, j].team == "black"
    for i in range(2, 6):
        for j in range(8):
            assert b.board[i, j].team is None
    assert b.turn == "white"


def test_update_board_base_move():
    b = Board()
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(b.board[3, 4], Pawn)
    assert b.board[3, 4].get_position() == (3, 4)
    assert b.board[3, 4].moved is True
    assert isinstance(b.board[1, 4], Dummy)
    assert b.turn == "black"


def test_update_board_enpasse_removes_captured_pawn():
    b = empty_board()
    b.board[4, 4] = Pawn((4, 4), "white")
    b.board[4, 3] = Pawn((4, 3), "black")
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert b.board[4, 4].team is None
    assert b.board[4, 3].team is None


def test_update_board_castling_kingside_moves_rook():
    b = empty_board()
    b.board[0, 4] = King((0, 4), "white")
    b.board[0, 7] = Rook((0, 7), "white")
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert b.board[0, 4].team is None
    assert b.board[0, 7].team is None


def test_update_board_evolve_promotes_to_queen():
    b = empty_board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].symbol == 'wq '
    assert b.board[6, 0].team is None


def test_board_copy_is_deep_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.board[0, 1] is not b.board[0, 1]
    assert c.equal(b)
    assert c.turn == b.turn
    c.update_board(((1, 0), (2, 0), "base"), "white")
    assert isinstance(b.board[1, 0], Pawn)
    assert not c.equal(b)
    assert not b.equal(c)


def test_board_str_layout(capsys):
    b = Board()
    text = str(b)
    lines = text.split("\n")
    assert lines[0] == "  A  B  C  D  E  F  G  H "
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")
    assert "wK " in text and "bK " in text
    b.view_board()
    out = capsys.readouterr().out
    assert "wp " in out


# --------------------------------------------------------------------- Knight

def test_knight_basic_attributes():
    w = Knight((0, 1), "white")
    k = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.symbol == 'wk '
    assert k.symbol == 'bk '
    assert w.get_position() == (0, 1)
    assert w.moved is False


def test_knight_center_has_eight_moves():
    b = empty_board()
    knight = Knight((3, 3), "white")
    b.board[3, 3] = knight
    moves = knight.get_moves([b])
    assert len(moves) == 8
    assert end_squares(moves) == {(1, 2), (1, 4), (2, 1), (4, 1),
                                  (5, 2), (5, 4), (4, 5), (2, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves():
    b = empty_board()
    knight = Knight((0, 0), "black")
    b.board[0, 0] = knight
    moves = knight.get_moves([b])
    assert end_squares(moves) == {(2, 1), (1, 2)}


def test_knight_blocked_by_own_and_captures_enemy():
    b = empty_board()
    knight = Knight((3, 3), "white")
    b.board[3, 3] = knight
    b.board[5, 4] = Pawn((5, 4), "white")
    b.board[5, 2] = Pawn((5, 2), "black")
    squares = end_squares(knight.get_moves([b]))
    assert (5, 4) not in squares
    assert (5, 2) in squares
    assert len(squares) == 7


def test_engine_initial_state_and_turn():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert engine.get_turn() == ("white", "human")
    assert len(engine.game_states) == 1
    assert engine.white_check is False
    assert engine.black_check is False
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    engine2 = GameEngine(Board(), "computer", "human")
    assert engine2.get_turn() == ("white", "computer")


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    assert end_squares(moves) == {(2, 0), (2, 2)}


def test_engine_num_alph_converts_to_chess_notation():
    result = GameEngine.num_alph({1: ((1, 4), (3, 4), "base"),
                                  2: ((0, 1), (2, 0), "base")})
    assert result == {1: ["E2", "E4"], 2: ["B1", "A3"]}


def test_engine_undo_move_restores_previous_state():
    engine = GameEngine(Board(), "computer", "computer")
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert isinstance(engine.Board.board[1, 4], Pawn)
    assert engine.Board.board[3, 4].team is None


def test_engine_choose_move_computer_applies_move():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert engine.get_turn() == ("black", "computer")
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert len(engine.game_states) == 2


def test_engine_choose_move_errors():
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    engine = GameEngine(Board(), "robot", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(((1, 4), (3, 4), "base"))
    engine = GameEngine(Board())
    with pytest.raises(ValueError):
        engine.choose_move(((1, 4), (3, 4), "base"))


def test_engine_is_check_detects_knight_attack():
    b = empty_board()
    b.board[0, 4] = King((0, 4), "white")
    b.board[7, 4] = King((7, 4), "black")
    b.board[2, 3] = Knight((2, 3), "black")
    engine = GameEngine(b)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_engine_compare_states():
    engine = GameEngine(Board())
    a = Board()
    b = Board()
    assert engine.compare_states(a, b)
    b.update_board(((1, 0), (2, 0), "base"), "white")
    assert not engine.compare_states(a, b)


def test_engine_stalemate_3states_exits_on_repeated_positions():
    engine = GameEngine(Board())
    engine.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_engine_stalemate_no_move_exits_when_team_has_no_moves():
    b = empty_board()
    b.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(b)
    assert engine.all_moves() == []
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_engine_all_moves_from_start_is_twenty():
    engine = GameEngine(Board())
    assert len(engine.all_moves()) == 20
    engine.is_stalemate_no_move("white")
