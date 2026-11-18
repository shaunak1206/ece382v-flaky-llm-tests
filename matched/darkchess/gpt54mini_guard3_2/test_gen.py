import pytest
import numpy as np

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].team is None
            assert board[i, j].get_position() == (i, j)


def test_board_default_start_positions_piece_symbols():
    board = Board()
    assert board.turn == "white"
    assert board.board[0, 0].symbol == "wr "
    assert board.board[0, 1].symbol == "wk "
    assert board.board[0, 2].symbol == "wb "
    assert board.board[0, 3].symbol == "wq "
    assert board.board[0, 4].symbol == "wK "
    assert board.board[1, 0].symbol == "wp "
    assert board.board[6, 0].symbol == "bp "
    assert board.board[7, 0].symbol == "br "
    assert board.board[7, 1].symbol == "bk "
    assert board.board[7, 2].symbol == "bb "
    assert board.board[7, 3].symbol == "bq "
    assert board.board[7, 4].symbol == "bK "


def test_board_str_has_coordinates_and_two_legend_rows():
    board = Board()
    s = str(board)
    assert s.startswith("  A  B  C  D  E  F  G  H \n")
    assert s.endswith("  A  B  C  D  E  F  G  H \n")
    for n in range(1, 9):
        assert f"{n} " in s
        assert f" {n}\n" in s


def test_board_update_board_base_moves_piece_and_toggles_turn():
    board = Board()
    start = (1, 0)
    end = (2, 0)
    board.update_board((start, end, "base"), "white")
    assert board.turn == "black"
    assert board.board[end].symbol == "wp "
    assert board.board[end].get_position() == end
    assert board.board[end].moved is True
    assert isinstance(board.board[start], Dummy)
    assert board.board[start].get_position() == start


def test_board_update_board_enpassant_clears_captured_pawn():
    board = clean_board()
    board[4, 4] = Pawn((4, 4), "white")
    board[4, 4].moved = True
    board[4, 5] = Pawn((4, 5), "black")
    board[4, 5].moved = True
    b = Board(board)
    b.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert b.board[5, 5].symbol == "wp "
    assert b.board[4, 4].team is None
    assert b.board[4, 5].team is None
    assert b.turn == "black"


def test_board_update_board_castling_kingside_moves_rook_and_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 7] = Rook((0, 7), "white")
    b = Board(board)
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert b.board[0, 6].symbol == "wK "
    assert b.board[0, 5].symbol == "wr "
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)
    assert b.board[0, 6].moved is True
    assert b.board[0, 5].moved is True


def test_board_update_board_castling_queenside_moves_rook_and_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 0] = Rook((0, 0), "white")
    b = Board(board)
    b.update_board(((0, 4), (0, 2), "castling"), "white")
    assert b.board[0, 2].symbol == "wK "
    assert b.board[0, 3].symbol == "wr "
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 0], Dummy)


def test_board_update_board_evolve_replaces_piece_with_queen():
    board = clean_board()
    board[6, 0] = Pawn((6, 0), "white")
    b = Board(board)
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert b.board[7, 0].symbol == "wq "
    assert isinstance(b.board[6, 0], Dummy)
    assert b.turn == "black"


def test_board_copy_is_deep_copy_and_equal_matches_original():
    board = Board()
    copied = board.copy()
    assert board.equal(copied)
    assert board is not copied
    assert board.board is not copied.board
    copied.board[1, 0] = Dummy((1, 0))
    assert not board.equal(copied)
    assert board.board[1, 0].symbol == "wp "


def test_board_equal_false_when_symbols_differ():
    b1 = Board()
    b2 = Board()
    b2.board[1, 0] = Dummy((1, 0))
    assert b1.equal(b1.copy())
    assert not b1.equal(b2)


def test_knight_get_moves_white_corner_blocked_by_same_team():
    board = clean_board()
    knight = Knight((0, 0), "white")
    board[0, 0] = knight
    board[1, 2] = Pawn((1, 2), "white")  # blocks one candidate
    moves = knight.get_moves([Board(board)])
    ends = {m[1] for m in moves}
    assert ends == {(2, 1)}


def test_knight_get_moves_black_center_has_all_legal_moves():
    board = clean_board()
    knight = Knight((3, 3), "black")
    board[3, 3] = knight
    moves = knight.get_moves([Board(board)])
    ends = {m[1] for m in moves}
    assert ends == {
        (1, 2), (1, 4), (2, 1), (2, 5),
        (4, 1), (4, 5), (5, 2), (5, 4)
    }


def test_knight_get_symbol_invalid_team_raises():
    knight = Knight((0, 0), "green")
    with pytest.raises(ValueError):
        knight.get_symbol()


def test_gameengine_get_turn_and_num_alph():
    board = Board()
    engine = GameEngine(board, player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    move_dict = {1: ((0, 1), (2, 2), "base"), 2: ((7, 7), (7, 5), "base")}
    assert engine.num_alph(move_dict) == {
        1: ["B1", "C3"],
        2: ["H8", "F8"],
    }


def test_gameengine_is_check_detects_knight_attack():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[7, 4] = King((7, 4), "black")
    board[2, 5] = Knight((2, 5), "black")  # attacks (0, 4)
    engine = GameEngine(Board(board))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_is_checkmate_calls_sys_exit_when_no_escape(monkeypatch):
    board = clean_board()
    board[0, 0] = King((0, 0), "white")
    board[7, 7] = King((7, 7), "black")
    board[0, 7] = Rook((0, 7), "black")  # delivers check on row 0
    engine = GameEngine(Board(board), player1="computer", player2="computer")
    engine.game_states = [engine.Board.copy()]
    engine.is_check()
    assert engine.white_check is True

    exit_calls = []

    def fake_exit(code):
        exit_calls.append(code)
        raise SystemExit(code)

    monkeypatch.setattr("darkchess.src.gamengine.sys.exit", fake_exit)

    with pytest.raises(SystemExit):
        engine.is_checkmate("white")

    assert exit_calls == [0]
