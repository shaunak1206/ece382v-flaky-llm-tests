import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.rook import Rook


# ---------------------------------------------------------------- utils.clean_board

def test_clean_board_is_8x8_of_dummies_with_no_team():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            cell = board[i, j]
            assert isinstance(cell, Dummy)
            assert cell.team is None
            assert cell.get_position() == (i, j)
    # each call must return a fresh array, not a shared one
    other = clean_board()
    assert other is not board
    other[0, 0] = Pawn((0, 0), "white")
    assert isinstance(board[0, 0], Dummy)


# ---------------------------------------------------------------- Board setup

def test_board_default_setup_has_32_pieces_in_standard_layout():
    b = Board()
    pieces = [b.board[i, j] for i in range(8) for j in range(8)
              if not isinstance(b.board[i, j], Dummy)]
    assert len(pieces) == 32
    for j in range(8):
        assert isinstance(b.board[1, j], Pawn)
        assert b.board[1, j].team == "white"
        assert isinstance(b.board[6, j], Pawn)
        assert b.board[6, j].team == "black"
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[7, 3], Queen)
    assert isinstance(b.board[0, 0], Rook)
    assert isinstance(b.board[0, 1], Knight)
    assert b.turn == "white"


def test_board_uses_given_starting_board():
    arr = clean_board()
    arr[3, 3] = Knight((3, 3), "white")
    b = Board(arr)
    assert b.board is arr
    assert isinstance(b.board[3, 3], Knight)
    assert b.turn == "white"


# ---------------------------------------------------------------- Board.update_board

def test_update_board_base_move_relocates_piece_and_flips_turn():
    b = Board()
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(b.board[3, 4], Pawn)
    assert b.board[3, 4].get_position() == (3, 4)
    assert b.board[3, 4].moved is True
    assert isinstance(b.board[1, 4], Dummy)
    assert b.turn == "black"
    b.update_board(((6, 4), (4, 4), None), "black")  # None keyword acts as base
    assert isinstance(b.board[4, 4], Pawn)
    assert b.turn == "white"


def test_update_board_enpasse_removes_captured_pawn():
    arr = clean_board()
    arr[4, 4] = Pawn((4, 4), "white")
    arr[4, 5] = Pawn((4, 5), "black")
    b = Board(arr)
    b.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert isinstance(b.board[5, 5], Pawn)
    assert b.board[5, 5].team == "white"
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 5], Dummy)


def test_update_board_castling_kingside_moves_rook():
    b = Board()
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert isinstance(b.board[0, 7], Dummy)
    assert isinstance(b.board[0, 4], Dummy)


def test_update_board_evolve_promotes_pawn_to_queen():
    arr = clean_board()
    arr[6, 0] = Pawn((6, 0), "white")
    b = Board(arr)
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


def test_update_board_invalid_turn_raises_value_error():
    b = Board()
    b.turn = "green"
    with pytest.raises(ValueError):
        b.update_board(((1, 0), (2, 0), "base"), "white")


# ---------------------------------------------------------------- Board.copy / equal / __str__

def test_board_copy_is_independent_of_original():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.turn == b.turn
    c.board[1, 0] = Dummy((1, 0))
    assert isinstance(b.board[1, 0], Pawn)


def test_board_equal_detects_differences():
    b1 = Board()
    b2 = Board()
    assert b1.equal(b2)
    b2.update_board(((1, 0), (2, 0), "base"), "white")
    assert not b1.equal(b2)


def test_board_str_has_header_eight_ranks_and_footer():
    text = str(Board())
    lines = text.splitlines()
    assert len(lines) == 10
    assert lines[0].strip() == "A  B  C  D  E  F  G  H"
    assert lines[-1].strip() == "A  B  C  D  E  F  G  H"
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")


# ---------------------------------------------------------------- Knight

def test_knight_name_value_and_symbols():
    white = Knight((0, 1), "white")
    black = Knight((7, 1), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.get_symbol() == "wk "
    assert black.get_symbol() == "bk "


def test_knight_moves_from_center_gives_eight_moves():
    arr = clean_board()
    arr[3, 3] = Knight((3, 3), "white")
    b = Board(arr)
    moves = arr[3, 3].get_moves([b])
    ends = {m[1] for m in moves}
    assert len(moves) == 8
    assert ends == {(1, 2), (1, 4), (5, 2), (5, 4),
                    (2, 1), (4, 1), (2, 5), (4, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_moves_from_corner_are_clipped_to_board():
    arr = clean_board()
    arr[0, 0] = Knight((0, 0), "white")
    b = Board(arr)
    ends = {m[1] for m in arr[0, 0].get_moves([b])}
    assert ends == {(1, 2), (2, 1)}


def test_knight_does_not_capture_own_team_but_captures_enemy():
    arr = clean_board()
    arr[0, 0] = Knight((0, 0), "white")
    arr[1, 2] = Knight((1, 2), "white")   # friendly: blocked
    arr[2, 1] = Knight((2, 1), "black")   # enemy: capturable
    b = Board(arr)
    ends = {m[1] for m in arr[0, 0].get_moves([b])}
    assert ends == {(2, 1)}


def test_knight_copy_preserves_state_but_is_new_object():
    k = Knight((3, 3), "black")
    k.moved = True
    c = k.copy()
    assert isinstance(c, Knight)
    assert c is not k
    assert c.get_position() == (3, 3)
    assert c.team == "black"
    assert c.moved is True


# ---------------------------------------------------------------- GameEngine

def test_gameengine_defaults():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")


def test_get_turn_alternates_with_move_counter():
    engine = GameEngine(Board(), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")
    engine.move_counter = 2
    assert engine.get_turn() == ("white", "human")


def test_num_alph_converts_positions_to_algebraic():
    move_dict = {1: ((1, 4), (3, 4), "base"), 2: ((0, 1), (2, 0), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["E2", "E4"], 2: ["B1", "A3"]}


def test_gameengine_get_moves_for_starting_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    ends = {m[1] for m in moves}
    # (1, 3) is blocked by a friendly pawn
    assert ends == {(2, 0), (2, 2)}


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)


def test_is_check_detects_knight_attacking_black_king():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[7, 4] = King((7, 4), "black")
    arr[5, 3] = Knight((5, 3), "white")  # attacks (7, 4)
    engine = GameEngine(Board(arr))
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_choose_move_in_computer_mode_requires_move():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
