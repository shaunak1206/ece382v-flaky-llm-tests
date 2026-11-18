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


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].get_position() == (i, j)
            assert board[i, j].team is None


def test_board_default_start_position_has_expected_pieces():
    b = Board()
    assert b.turn == "white"
    assert isinstance(b.board[0, 0], Rook)
    assert isinstance(b.board[0, 1], Knight)
    assert isinstance(b.board[0, 3], Queen)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[1, 0], Pawn)
    assert b.board[1, 0].team == "white"
    assert isinstance(b.board[6, 0], Pawn)
    assert b.board[6, 0].team == "black"
    assert isinstance(b.board[7, 7], Rook)
    assert b.board[7, 7].team == "black"


def test_board_copy_is_deep_and_independent():
    b1 = Board()
    b2 = b1.copy()

    assert b1 is not b2
    assert b1.equal(b2)

    # Mutating copy should not affect original
    b2.board[1, 0].team = "black"
    assert b1.board[1, 0].team == "white"
    assert not b1.equal(b2)


def test_board_equal_detects_different_piece_layout():
    b1 = Board()
    b2 = Board()
    assert b1.equal(b2)

    b2.board[1, 0] = Dummy((1, 0))
    assert not b1.equal(b2)


def test_board_update_board_base_moves_piece_and_switches_turn():
    b = Board()
    start = (1, 0)
    end = (2, 0)
    piece = b.board[start]

    b.update_board((start, end, "base"), "white")

    assert isinstance(b.board[end], Pawn)
    assert b.board[end].team == "white"
    assert b.board[end].get_position() == end
    assert isinstance(b.board[start], Dummy)
    assert b.board[end].moved is True
    assert b.turn == "black"
    assert b.board[end] is piece


def test_board_update_board_enpassant_clears_captured_square():
    b = Board()
    b.board = clean_board()
    white_pawn = Pawn((4, 4), "white")
    black_pawn = Pawn((4, 5), "black")
    b.board[4, 4] = white_pawn
    b.board[4, 5] = black_pawn
    b.turn = "white"

    b.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert isinstance(b.board[5, 5], Pawn)
    assert b.board[5, 5].team == "white"
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 5], Dummy)
    assert b.turn == "black"


def test_board_update_board_castling_moves_king_and_rook_kingside():
    b = Board()
    b.board = clean_board()
    king = King((0, 4), "white")
    rook = Rook((0, 7), "white")
    b.board[0, 4] = king
    b.board[0, 7] = rook
    b.turn = "white"

    b.update_board(((0, 4), (0, 6), "castling"), "white")

    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)
    assert b.board[0, 6].get_position() == (0, 6)
    assert b.board[0, 5].get_position() == (0, 5)
    assert b.turn == "black"


def test_board_update_board_evolve_promotes_to_queen():
    b = Board()
    b.board = clean_board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.turn = "white"

    b.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)
    assert b.turn == "black"


def test_board_update_board_invalid_turn_raises_value_error():
    b = Board()
    b.turn = "invalid"
    with pytest.raises(ValueError):
        b.update_board(((1, 0), (2, 0), "base"), "white")


def test_knight_get_symbol_and_copy():
    k_white = Knight((3, 3), "white")
    k_black = Knight((3, 3), "black")

    assert k_white.get_symbol() == "wk "
    assert k_black.get_symbol() == "bk "

    k_white.moved = True
    k_copy = k_white.copy()
    assert isinstance(k_copy, Knight)
    assert k_copy is not k_white
    assert k_copy.get_position() == (3, 3)
    assert k_copy.team == "white"
    assert k_copy.moved is True


def test_knight_get_moves_from_center_has_eight_moves_on_empty_board():
    b = Board()
    b.board = clean_board()
    knight = Knight((3, 3), "white")
    b.board[3, 3] = knight

    moves = knight.get_moves([b])

    expected_ends = {
        (1, 2), (1, 4), (2, 1), (4, 1),
        (5, 2), (5, 4), (4, 5), (2, 5),
    }
    assert len(moves) == 8
    assert {m[1] for m in moves} == expected_ends
    assert all(m[0] == (3, 3) for m in moves)
    assert all(m[2] == "base" for m in moves)


def test_knight_get_moves_blocks_friendly_piece_and_allows_enemy_capture():
    b = Board()
    b.board = clean_board()
    knight = Knight((3, 3), "white")
    b.board[3, 3] = knight
    b.board[5, 4] = Pawn((5, 4), "white")  # friendly, should block
    b.board[4, 5] = Pawn((4, 5), "black")  # enemy, should be allowed

    moves = knight.get_moves([b])
    ends = {m[1] for m in moves}

    assert (5, 4) not in ends
    assert (4, 5) in ends


def test_gameengine_get_turn_and_num_alph():
    b = Board()
    engine = GameEngine(b, player1="human", player2="computer")

    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    move_dict = {1: ((0, 1), (2, 2), "base")}
    assert engine.num_alph(move_dict) == {1: ["B1", "C3"]}


def test_gameengine_all_moves_and_get_moves_work_for_knight():
    b = Board()
    b.board = clean_board()
    knight = Knight((3, 3), "white")
    b.board[3, 3] = knight

    engine = GameEngine(b)
    moves = engine.get_moves((3, 3))
    assert len(moves) == 8
    assert set(engine.all_moves()) == set(moves)


def test_gameengine_is_check_detects_check_and_stalemate_no_move_exits():
    b = Board()
    b.board = clean_board()
    b.board[0, 4] = King((0, 4), "white")
    b.board[7, 4] = King((7, 4), "black")
    b.board[1, 4] = Rook((1, 4), "black")  # attacks white king
    engine = GameEngine(b)

    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False

    # Create a board where the side to move has no legal moves and is not in check
    b2 = Board()
    b2.board = clean_board()
    b2.board[0, 0] = King((0, 0), "white")
    b2.board[7, 7] = King((7, 7), "black")
    engine2 = GameEngine(b2)
    engine2.move_counter = 1  # black to move
    with pytest.raises(SystemExit):
        engine2.is_stalemate_no_move("black")
