import sys
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board

from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.bishop import Bishop
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    assert board.dtype.name == "object" or board.dtype is not None
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].get_position() == (i, j)


def test_board_default_starting_position_has_expected_pieces():
    board = Board()
    assert isinstance(board.board[0, 0], Rook)
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 2], Bishop)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[1, 0], Pawn)
    assert isinstance(board.board[6, 0], Pawn)
    assert isinstance(board.board[7, 4], King)
    assert board.turn == "white"


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    s = str(board)
    assert "A  B  C  D  E  F  G  H" in s
    assert "1 " in s
    assert "8 " in s
    assert "wk " in s  # white knight
    assert "bk " in s  # black knight


def test_board_copy_is_independent_from_original():
    board = Board()
    board_copy = board.copy()

    assert board.equal(board_copy)
    assert board is not board_copy
    assert board.board is not board_copy.board

    board_copy.board[1, 0] = Dummy((1, 0))
    assert not board.equal(board_copy)
    assert isinstance(board.board[1, 0], Pawn)


def test_board_equal_detects_same_and_different_boards():
    board1 = Board()
    board2 = Board()
    assert board1.equal(board2)

    board2.board[1, 0] = Dummy((1, 0))
    assert not board1.equal(board2)


def test_board_update_base_move_updates_positions_and_turn():
    board = Board()
    move = ((1, 0), (2, 0), "base")
    board.update_board(move, "white")

    assert isinstance(board.board[2, 0], Pawn)
    assert isinstance(board.board[1, 0], Dummy)
    assert board.board[2, 0].get_position() == (2, 0)
    assert board.board[2, 0].moved is True
    assert board.turn == "black"


def test_board_update_en_passant_removes_captured_pawn():
    board = clean_board()
    board[4, 4] = Pawn((4, 4), "white")
    board[4, 5] = Pawn((4, 5), "black")
    b = Board(board)

    b.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert isinstance(b.board[5, 5], Pawn)
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 5], Dummy)
    assert b.board[5, 5].team == "white"


def test_board_update_castling_moves_rook_kingside():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 7] = Rook((0, 7), "white")
    b = Board(board)

    b.update_board(((0, 4), (0, 6), "castling"), "white")

    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)
    assert b.board[0, 6].get_position() == (0, 6)
    assert b.board[0, 5].get_position() == (0, 5)


def test_board_update_castling_moves_rook_queenside():
    board = clean_board()
    board[7, 4] = King((7, 4), "black")
    board[7, 0] = Rook((7, 0), "black")
    b = Board(board)
    b.turn = "black"

    b.update_board(((7, 4), (7, 2), "castling"), "black")

    assert isinstance(b.board[7, 2], King)
    assert isinstance(b.board[7, 3], Rook)
    assert isinstance(b.board[7, 4], Dummy)
    assert isinstance(b.board[7, 0], Dummy)


def test_board_update_evolve_replaces_pawn_with_queen():
    board = clean_board()
    board[6, 0] = Pawn((6, 0), "white")
    b = Board(board)

    b.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)
    assert b.turn == "black"


def test_knight_get_moves_center_generates_eight_moves():
    board = clean_board()
    k = Knight((3, 3), "white")
    board[3, 3] = k
    b = Board(board)

    moves = k.get_moves([b])
    expected = {
        ((3, 3), (1, 2), "base"),
        ((3, 3), (1, 4), "base"),
        ((3, 3), (2, 1), "base"),
        ((3, 3), (4, 1), "base"),
        ((3, 3), (5, 2), "base"),
        ((3, 3), (5, 4), "base"),
        ((3, 3), (4, 5), "base"),
        ((3, 3), (2, 5), "base"),
    }
    assert set(moves) == expected


def test_knight_get_moves_respects_board_edges():
    board = clean_board()
    k = Knight((0, 0), "white")
    board[0, 0] = k
    b = Board(board)

    moves = k.get_moves([b])
    assert set(moves) == {
        ((0, 0), (1, 2), "base"),
        ((0, 0), (2, 1), "base"),
    }


def test_knight_get_moves_does_not_include_friendly_occupied_squares():
    board = clean_board()
    k = Knight((3, 3), "white")
    board[3, 3] = k
    board[1, 2] = Pawn((1, 2), "white")
    board[2, 5] = Pawn((2, 5), "white")
    b = Board(board)

    moves = k.get_moves([b])
    destinations = {m[1] for m in moves}
    assert (1, 2) not in destinations
    assert (2, 5) not in destinations
    assert (1, 4) in destinations


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

    move_dict = {1: ((0, 0), (2, 1), "base")}
    assert engine.num_alph(move_dict) == {1: ["A1", "B3"]}


def test_gameengine_is_check_detects_attack_on_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[7, 4] = King((7, 4), "black")
    board[1, 4] = Rook((1, 4), "black")
    b = Board(board)
    engine = GameEngine(b)

    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_compare_states_uses_board_equality():
    engine = GameEngine(Board())
    s = engine.game_states[0]
    t = engine.game_states[0].copy()
    assert engine.compare_states(s, t) is True
    t.board[1, 0] = Dummy((1, 0))
    assert engine.compare_states(s, t) is False


def test_gameengine_all_moves_returns_current_team_moves():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[7, 4] = King((7, 4), "black")
    board[1, 0] = Knight((1, 0), "white")
    b = Board(board)
    engine = GameEngine(b)

    moves = engine.all_moves()
    assert any(move[0] == (1, 0) for move in moves)
    assert any(move[0] == (0, 4) for move in moves)
    assert all(b.board[m[0]].team == "white" for m in moves)
