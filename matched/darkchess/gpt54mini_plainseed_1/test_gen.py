import pytest
import numpy as np

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.pawn import Pawn
from darkchess.src.bishop import Bishop


def make_empty_board():
    return Board(clean_board())


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].pos == (i, j)


def test_board_default_start_position_has_expected_pieces():
    board = Board()
    assert isinstance(board.board[0, 0], Rook)
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 2], Bishop)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[1, 0], Pawn)
    assert isinstance(board.board[6, 0], Pawn)
    assert isinstance(board.board[7, 7], Rook)
    assert board.turn == "white"


def test_board_copy_creates_independent_board():
    board = Board()
    board_copy = board.copy()

    assert board.equal(board_copy)
    assert board is not board_copy
    assert board.board is not board_copy.board

    board_copy.board[0, 0] = Dummy((0, 0), None)
    assert not board.equal(board_copy)


def test_board_equal_detects_different_piece_symbols():
    b1 = Board(clean_board())
    b2 = Board(clean_board())
    assert b1.equal(b2)

    b2.board[3, 3] = Knight((3, 3), "white")
    assert not b1.equal(b2)


def test_board_update_board_base_moves_piece_and_switches_turn():
    board = Board(clean_board())
    knight = Knight((0, 1), "white")
    board.board[0, 1] = knight

    board.update_board(((0, 1), (2, 2), "base"), "white")

    assert isinstance(board.board[0, 1], Dummy)
    assert isinstance(board.board[2, 2], Knight)
    assert board.board[2, 2].pos == (2, 2)
    assert board.board[2, 2].moved is True
    assert board.turn == "black"


def test_board_update_board_enpassant_clears_captured_pawn():
    board = Board(clean_board())
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 5] = Pawn((4, 5), "black")

    board.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert isinstance(board.board[5, 5], Pawn)
    assert board.board[5, 5].team == "white"
    assert board.turn == "black"


def test_board_update_board_castling_king_side_moves_rook_too():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")

    board.update_board(((0, 4), (0, 6), "castling"), "white")

    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert isinstance(board.board[0, 7], Dummy)
    assert board.board[0, 6].moved is True
    assert board.board[0, 5].moved is True
    assert board.turn == "black"


def test_board_update_board_castling_queen_side_moves_rook_too():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 0] = Rook((0, 0), "white")

    board.update_board(((0, 4), (0, 2), "castling"), "white")

    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 2], King)
    assert isinstance(board.board[0, 3], Rook)
    assert isinstance(board.board[0, 0], Dummy)
    assert board.turn == "black"


def test_board_update_board_evolve_promotes_to_queen():
    board = Board(clean_board())
    board.board[6, 0] = Pawn((6, 0), "white")

    board.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(board.board[6, 0], Dummy)
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert board.turn == "black"


def test_board_update_board_invalid_turn_raises_value_error():
    board = Board(clean_board())
    board.turn = "neither"
    board.board[0, 1] = Knight((0, 1), "white")

    with pytest.raises(ValueError):
        board.update_board(((0, 1), (2, 2), "base"), "white")


def test_knight_get_symbol_for_both_teams():
    assert Knight((0, 0), "white").get_symbol() == "wk "
    assert Knight((0, 0), "black").get_symbol() == "bk "


def test_knight_get_moves_on_empty_board_from_center():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight

    moves = knight.get_moves([board])

    expected_ends = {
        (1, 2), (1, 4), (2, 1), (2, 5),
        (4, 1), (4, 5), (5, 2), (5, 4),
    }
    assert {m[1] for m in moves} == expected_ends
    assert all(m[0] == (3, 3) for m in moves)
    assert all(m[2] == "base" for m in moves)


def test_knight_get_moves_blocks_friendly_occupied_squares_only():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    board.board[1, 2] = Pawn((1, 2), "white")   # friendly, should be blocked
    board.board[1, 4] = Pawn((1, 4), "black")   # enemy, should be allowed

    moves = knight.get_moves([board])
    ends = {m[1] for m in moves}

    assert (1, 2) not in ends
    assert (1, 4) in ends


def test_knight_get_moves_from_corner_respects_bounds():
    board = Board(clean_board())
    knight = Knight((0, 0), "white")
    board.board[0, 0] = knight

    moves = knight.get_moves([board])
    assert {m[1] for m in moves} == {(1, 2), (2, 1)}


def test_gameengine_get_turn_and_num_alph():
    board = Board(clean_board())
    engine = GameEngine(board, player1="human", player2="computer")

    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    converted = engine.num_alph({1: ((0, 0), (7, 7), "base")})
    assert converted == {1: ["A1", "H8"]}


def test_gameengine_all_moves_and_is_check_with_simple_position():
    board = Board(clean_board())
    board.board[0, 0] = King((0, 0), "white")
    board.board[7, 7] = King((7, 7), "black")
    board.board[6, 6] = Knight((6, 6), "white")  # attacks 7,4 and 5,4 etc., not king
    board.board[5, 6] = Knight((5, 6), "black")  # attacks 7,7

    engine = GameEngine(board)

    moves = engine.all_moves()
    assert isinstance(moves, list)
    assert len(moves) > 0

    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False
