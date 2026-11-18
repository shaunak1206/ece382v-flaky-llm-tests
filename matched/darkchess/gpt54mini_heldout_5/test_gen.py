import sys
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].team is None
            assert board[i, j].get_position() == (i, j)


def test_board_default_starting_position_piece_symbols():
    board = Board()
    assert board.turn == "white"
    assert board.board[0, 0].symbol == "wr "
    assert board.board[0, 1].symbol == "wk "
    assert board.board[0, 2].symbol == "wb "
    assert board.board[0, 3].symbol == "wq "
    assert board.board[0, 4].symbol == "wK "
    assert board.board[7, 4].symbol == "bK "
    assert board.board[6, 0].symbol == "bp "
    assert board.board[1, 7].symbol == "wp "


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    rendered = str(board)
    assert "A  B  C  D  E  F  G  H" in rendered
    assert "8 " in rendered
    assert "1 " in rendered
    assert "wk " in rendered
    assert "bK " in rendered


def test_board_copy_is_deep_and_preserves_turn():
    board = Board()
    copied = board.copy()

    assert copied is not board
    assert copied.equal(board)
    assert copied.turn == board.turn

    copied.board[0, 0] = Dummy((0, 0))
    assert board.board[0, 0].symbol == "wr "
    assert copied.board[0, 0].symbol != board.board[0, 0].symbol


def test_board_equal_detects_difference():
    board1 = Board()
    board2 = board1.copy()
    assert board1.equal(board2) is True

    board2.board[0, 0] = Dummy((0, 0))
    assert board1.equal(board2) is False


def test_update_board_base_moves_piece_and_flips_turn():
    board = Board()
    move = ((1, 0), (2, 0), "base")
    board.update_board(move, "white")

    assert isinstance(board.board[1, 0], Dummy)
    assert board.board[2, 0].symbol == "wp "
    assert board.board[2, 0].get_position() == (2, 0)
    assert board.board[2, 0].moved is True
    assert board.turn == "black"


def test_update_board_enpassant_captures_target_and_flips_turn():
    board = Board()
    board.board = clean_board()
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 5] = Pawn((4, 5), "black")
    board.board[4, 5].moved = True

    board.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert board.board[5, 5].symbol == "wp "
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert board.turn == "black"


def test_update_board_castling_queenside_moves_rook_too():
    board = Board()
    board.board = clean_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 0] = Rook((0, 0), "white")
    board.board[0, 3] = Dummy((0, 3))
    board.board[0, 2] = Dummy((0, 2))

    board.update_board(((0, 4), (0, 2), "castling"), "white")

    assert board.board[0, 2].symbol == "wK "
    assert board.board[0, 3].symbol == "wr "
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 0], Dummy)


def test_update_board_castling_kingside_moves_rook_too():
    board = Board()
    board.board = clean_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")
    board.board[0, 5] = Dummy((0, 5))
    board.board[0, 6] = Dummy((0, 6))

    board.update_board(((0, 4), (0, 6), "castling"), "white")

    assert board.board[0, 6].symbol == "wK "
    assert board.board[0, 5].symbol == "wr "
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 7], Dummy)


def test_update_board_evolve_promotes_to_queen_and_flips_turn():
    board = Board()
    board.board = clean_board()
    board.board[6, 0] = Pawn((6, 0), "white")

    board.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert board.board[7, 0].symbol == "wq "
    assert isinstance(board.board[6, 0], Dummy)
    assert board.turn == "black"


def test_knight_get_symbol_and_copy():
    k = Knight((3, 3), "white")
    k.moved = True
    copied = k.copy()

    assert k.get_name() == "knight"
    assert k.get_value() == 3
    assert k.get_symbol() == "wk "
    assert copied is not k
    assert copied.get_position() == (3, 3)
    assert copied.team == "white"
    assert copied.moved is True


def test_knight_get_moves_from_center_without_blockers():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    engine = GameEngine(board)

    moves = knight.get_moves(engine.game_states)
    destinations = {m[1] for m in moves}
    expected = {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
    assert destinations == expected
    assert all(m[2] == "base" for m in moves)


def test_knight_get_moves_blocks_friendly_pieces_and_allows_enemy():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    board.board[1, 2] = Pawn((1, 2), "white")  # friendly block
    board.board[1, 4] = Pawn((1, 4), "black")  # enemy capture allowed
    engine = GameEngine(board)

    moves = knight.get_moves(engine.game_states)
    destinations = {m[1] for m in moves}

    assert (1, 2) not in destinations
    assert (1, 4) in destinations


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

    move_dict = {1: ((0, 0), (7, 7), "base")}
    assert engine.num_alph(move_dict) == {1: ["A1", "H8"]}


def test_gameengine_compare_states_and_undo_move():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert engine.compare_states(original, engine.game_states[0]) is True

    engine.undo_move()
    assert engine.Board.equal(original)
    assert len(engine.game_states) == 1


def test_gameengine_is_check_detects_simple_check():
    board = Board()
    board.board = clean_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    board.board[6, 4] = Rook((6, 4), "white")

    engine = GameEngine(board)
    engine.is_check()

    assert engine.black_check is True
    assert engine.white_check is False
