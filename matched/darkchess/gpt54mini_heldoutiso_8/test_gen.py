import sys
import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.queen import Queen
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.bishop import Bishop
from darkchess.src.king import King


class FakePiece:
    def __init__(self, pos, team, symbol="fp "):
        self.pos = pos
        self.team = team
        self.symbol = symbol
        self.moved = False

    def set_position(self, pos):
        self.pos = pos

    def get_position(self):
        return self.pos

    def copy(self):
        cp = FakePiece(self.pos, self.team, self.symbol)
        cp.moved = self.moved
        return cp

    def get_moves(self, game_states):
        return []

    def get_symbol(self):
        return self.symbol


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    assert board.dtype == object or board.dtype.name == "object"
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].team is None


def test_board_default_setup_contains_expected_pieces():
    board = Board()
    assert board.turn == "white"
    assert isinstance(board.board[0, 0], Rook)
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 2], Bishop)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[1, 0], Pawn)
    assert isinstance(board.board[7, 4], King)
    assert board.board[0, 0].team == "white"
    assert board.board[7, 4].team == "black"


def test_board_str_has_coordinates_and_piece_symbols():
    board = Board()
    s = str(board)
    assert "A  B  C  D  E  F  G  H" in s
    assert "8 " in s and "1 " in s
    assert "wk " in s
    assert "bk " in s


def test_board_copy_is_deep_and_preserves_turn():
    board = Board()
    copied = board.copy()
    assert copied is not board
    assert copied.turn == board.turn
    assert copied.equal(board)
    copied.board[0, 0].symbol = "xx "
    assert board.board[0, 0].symbol != "xx "
    assert not copied.equal(board)


def test_board_equal_detects_difference():
    b1 = Board()
    b2 = Board()
    assert b1.equal(b2)
    b2.board[0, 0] = Dummy((0, 0))
    assert not b1.equal(b2)


def test_update_board_base_moves_piece_and_flips_turn():
    board = Board()
    start = (1, 0)
    end = (2, 0)
    piece = board.board[start]
    board.update_board((start, end, "base"), "white")
    assert isinstance(board.board[start], Dummy)
    assert board.board[end] is piece
    assert board.board[end].pos == end
    assert board.board[end].moved is True
    assert board.turn == "black"


def test_update_board_enpassant_clears_captured_pawn():
    board = Board()
    board.board = clean_board()
    mover = FakePiece((4, 4), "white", "wp ")
    captured = FakePiece((4, 5), "black", "bp ")
    board.board[4, 4] = mover
    board.board[4, 5] = captured
    board.turn = "white"

    board.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert board.board[5, 5] is mover
    assert board.board[5, 5].pos == (5, 5)
    assert board.turn == "black"


def test_update_board_castling_king_side_moves_rook_and_king():
    board = Board()
    board.board = clean_board()
    king = FakePiece((0, 4), "white", "wK ")
    rook = FakePiece((0, 7), "white", "wR ")
    board.board[0, 4] = king
    board.board[0, 7] = rook
    board.turn = "white"

    board.update_board(((0, 4), (0, 6), "castling"), "white")

    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 7], Dummy)
    assert board.board[0, 6] is king
    assert board.board[0, 5] is rook
    assert board.board[0, 6].pos == (0, 6)
    assert board.board[0, 5].pos == (0, 5)
    assert board.turn == "black"


def test_update_board_castling_queen_side_moves_rook_and_king():
    board = Board()
    board.board = clean_board()
    king = FakePiece((7, 4), "black", "bK ")
    rook = FakePiece((7, 0), "black", "bR ")
    board.board[7, 4] = king
    board.board[7, 0] = rook
    board.turn = "black"

    board.update_board(((7, 4), (7, 2), "castling"), "black")

    assert isinstance(board.board[7, 4], Dummy)
    assert isinstance(board.board[7, 0], Dummy)
    assert board.board[7, 2] is king
    assert board.board[7, 3] is rook
    assert board.turn == "white"


def test_update_board_evolve_creates_queen():
    board = Board()
    board.board = clean_board()
    pawn = FakePiece((6, 0), "white", "wp ")
    board.board[6, 0] = pawn
    board.turn = "white"

    board.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(board.board[6, 0], Dummy)
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert board.turn == "black"


def test_knight_get_symbol_and_copy():
    k = Knight((3, 3), "white")
    k.moved = True
    cp = k.copy()
    assert k.get_symbol() == "wk "
    assert cp is not k
    assert isinstance(cp, Knight)
    assert cp.pos == k.pos
    assert cp.team == k.team
    assert cp.moved is True


def test_knight_get_moves_from_center_blocks_friendly_pieces():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    board[5, 4] = FakePiece((5, 4), "white", "wp ")  # friendly block
    board[2, 1] = FakePiece((2, 1), "black", "bp ")  # enemy capture allowed

    moves = knight.get_moves([type("State", (), {"board": board})()])
    ends = {m[1] for m in moves}
    assert (5, 4) not in ends
    assert (2, 1) in ends
    assert len(ends) == len(moves)


def test_knight_get_moves_from_corner_has_two_moves():
    board = clean_board()
    knight = Knight((0, 0), "black")
    board[0, 0] = knight

    moves = knight.get_moves([type("State", (), {"board": board})()])
    assert set(m[1] for m in moves) == {(1, 2), (2, 1)}
    assert all(m[2] == "base" for m in moves)


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board(), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    move_dict = {1: ((0, 0), (7, 7), "base")}
    assert engine.num_alph(move_dict) == {1: ["A1", "H8"]}


def test_gameengine_is_check_sets_flags_for_simple_check():
    board = Board()
    board.board = clean_board()
    white_king = FakePiece((0, 4), "white", "wK ")
    black_attacker = FakePiece((7, 4), "black", "bk ")
    board.board[0, 4] = white_king
    board.board[7, 4] = black_attacker

    def white_moves(game_states):
        return [((7, 4), (0, 4), "base")]

    def black_moves(game_states):
        return []

    black_attacker.get_moves = white_moves
    white_king.get_moves = black_moves

    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_undo_move_restores_previous_state():
    board = Board()
    engine = GameEngine(board)
    original = engine.Board.copy()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()
    assert engine.Board.equal(original)
    assert len(engine.game_states) == 1


def test_gameengine_compare_states_uses_board_equal():
    engine = GameEngine(Board())
    s = engine.Board.copy()
    t = engine.Board.copy()
    assert engine.compare_states(s, t) is True
    t.board[0, 0] = Dummy((0, 0))
    assert engine.compare_states(s, t) is False
