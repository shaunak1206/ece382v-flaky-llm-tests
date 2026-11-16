import pytest
import numpy as np  # noqa: F401  (project dependency; used indirectly by the code under test)

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen


def _board_with(pieces):
    """Build a Board from an empty clean_board with the given (pos, piece) pairs placed."""
    arr = clean_board()
    for pos, piece in pieces:
        arr[pos] = piece
    return Board(arr)


# ---------------------------------------------------------------- clean_board

def test_clean_board_is_8x8_of_positioned_dummies():
    arr = clean_board()
    assert arr.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(arr[i, j], Dummy)
            assert arr[i, j].get_position() == (i, j)


# ---------------------------------------------------------------- Board setup

def test_default_board_placement_and_piece_count():
    b = Board()
    assert b.turn == "white"
    assert isinstance(b.board[0, 4], King) and b.board[0, 4].team == "white"
    assert isinstance(b.board[7, 3], Queen) and b.board[7, 3].team == "black"
    assert isinstance(b.board[1, 0], Pawn) and b.board[1, 0].team == "white"
    assert isinstance(b.board[6, 7], Pawn) and b.board[6, 7].team == "black"
    assert isinstance(b.board[0, 1], Knight)
    pieces = [b.board[i, j] for i in range(8) for j in range(8)
              if not isinstance(b.board[i, j], Dummy)]
    assert len(pieces) == 32


def test_board_uses_given_starting_board_object():
    arr = clean_board()
    b = Board(arr)
    assert b.board is arr
    assert b.turn == "white"


# ---------------------------------------------------------------- update_board

def test_update_board_base_move_moves_piece_and_leaves_dummy():
    b = _board_with([((0, 1), Knight((0, 1), "white"))])
    b.update_board(((0, 1), (2, 0), "base"), "white")
    assert isinstance(b.board[2, 0], Knight)
    assert b.board[2, 0].get_position() == (2, 0)
    assert b.board[2, 0].moved is True
    assert isinstance(b.board[0, 1], Dummy)
    assert b.turn == "black"


def test_update_board_turn_alternates():
    b = Board()
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert b.turn == "black"
    b.update_board(((6, 4), (4, 4), "base"), "black")
    assert b.turn == "white"


def test_update_board_unknown_turn_raises_value_error():
    b = Board()
    b.turn = "green"
    with pytest.raises(ValueError):
        b.update_board(((0, 1), (2, 0), "base"), "white")


def test_castling_kingside_moves_king_and_rook():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[0, 7] = Rook((0, 7), "white")
    b = Board(arr)
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert b.board[0, 6].get_position() == (0, 6)
    assert b.board[0, 6].moved is True
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert b.board[0, 5].moved is True
    assert isinstance(b.board[0, 7], Dummy)
    assert isinstance(b.board[0, 4], Dummy)


def test_castling_queenside_moves_rook_to_d_file():
    arr = clean_board()
    arr[0, 4] = King((0, 4), "white")
    arr[0, 0] = Rook((0, 0), "white")
    b = Board(arr)
    b.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(b.board[0, 2], King)
    assert isinstance(b.board[0, 3], Rook)
    assert b.board[0, 3].get_position() == (0, 3)
    assert isinstance(b.board[0, 0], Dummy)


def test_enpassant_removes_captured_pawn():
    arr = clean_board()
    arr[4, 4] = Pawn((4, 4), "white")
    arr[4, 5] = Pawn((4, 5), "black")
    b = Board(arr)
    b.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert isinstance(b.board[5, 5], Pawn)
    assert b.board[5, 5].team == "white"
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 5], Dummy)


def test_evolve_replaces_pawn_with_queen():
    arr = clean_board()
    arr[6, 0] = Pawn((6, 0), "white")
    b = Board(arr)
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


# ---------------------------------------------------------------- copy / equal / str

def test_board_copy_is_independent_of_original():
    b = Board()
    c = b.copy()
    for i in range(8):
        for j in range(8):
            assert c.board[i, j] is not b.board[i, j]
            assert c.board[i, j].get_position() == (i, j)
    c.board[1, 0] = Dummy((1, 0))
    c.turn = "black"
    assert isinstance(b.board[1, 0], Pawn)
    assert b.turn == "white"
    assert b.equal(Board())
    assert not b.equal(c)


def test_equal_detects_difference_and_is_symmetric():
    b1 = Board()
    assert b1.equal(Board())
    b2 = Board()
    b2.update_board(((1, 4), (3, 4), "base"), "white")
    assert not b1.equal(b2)
    assert not b2.equal(b1)


def test_str_layout_is_stable():
    s = str(Board())
    lines = s.split("\n")
    assert len(lines) == 11
    assert lines[0] == "  A  B  C  D  E  F  G  H "
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")
    assert lines[9] == "  A  B  C  D  E  F  G  H "
    assert lines[10] == ""
    assert "wk " in s and "bk " in s


# ---------------------------------------------------------------- Knight

def test_knight_name_value_and_symbols():
    assert Knight((0, 0), "white").get_name() == "knight"
    assert Knight((0, 0), "white").get_value() == 3
    assert Knight((0, 0), "white").get_symbol() == "wk "
    assert Knight((0, 0), "black").get_symbol() == "bk "
    with pytest.raises(ValueError):
        Knight((0, 0), "red").get_symbol()


def test_knight_moves_from_center():
    k = Knight((3, 3), "white")
    b = _board_with([((3, 3), k)])
    moves = k.get_moves([b])
    ends = {m[1] for m in moves}
    expected = {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
    assert len(moves) == 8
    assert ends == expected
    for m in moves:
        assert m[0] == (3, 3)
        assert m[2] == "base"


def test_knight_moves_from_corner():
    k = Knight((0, 0), "white")
    b = _board_with([((0, 0), k)])
    ends = {m[1] for m in k.get_moves([b])}
    assert ends == {(2, 1), (1, 2)}


def test_knight_blocked_by_own_piece_but_can_capture_enemy():
    k = Knight((3, 3), "white")
    b = _board_with([
        ((3, 3), k),
        ((5, 4), Knight((5, 4), "white")),   # own piece: must be excluded
        ((1, 2), Knight((1, 2), "black")),   # enemy piece: must be included
    ])
    ends = {m[1] for m in k.get_moves([b])}
    assert (5, 4) not in ends
    assert (1, 2) in ends
    assert len(ends) == 7


def test_knight_moves_invalid_team_raises():
    k = Knight((3, 3), "red")
    with pytest.raises(ValueError):
        k.get_moves([Board(clean_board())])


# ---------------------------------------------------------------- GameEngine

def test_engine_initial_state_and_get_turn():
    e = GameEngine(Board())
    assert e.move_counter == 0
    assert len(e.game_states) == 1
    assert e.player1 == "human" and e.player2 == "human"
    assert e.white_check is False and e.black_check is False
    assert e.get_turn() == ("white", "human")
    e2 = GameEngine(Board(), player1="computer", player2="computer")
    e2.move_counter = 1
    assert e2.get_turn() == ("black", "computer")


def test_num_alph_converts_moves_to_algebraic():
    result = GameEngine.num_alph({1: ((1, 4), (3, 4), "base")})
    assert result == {1: ["E2", "E4"]}


def test_all_moves_contains_knight_move_and_only_white_starts():
    e = GameEngine(Board())
    moves = e.all_moves()
    assert ((0, 1), (2, 0), "base") in moves
    for m in moves:
        assert e.Board.board[m[0]].team == "white"


def test_stalemate_no_move_exits_when_team_has_no_pieces():
    arr = clean_board()
    arr[7, 1] = Knight((7, 1), "black")
    e = GameEngine(Board(arr))
    with pytest.raises(SystemExit) as exc:
        e.is_stalemate_no_move("white")
    assert exc.value.code == 0


def test_stalemate_3states_exits_on_repeated_states():
    e = GameEngine(Board())
    for _ in range(8):
        e.game_states.append(e.Board.copy())
    assert len(e.game_states) == 9
    with pytest.raises(SystemExit) as exc:
        e.is_stalemate_3states("white")
    assert exc.value.code == 0


def test_undo_move_restores_previous_board():
    b = Board()
    e = GameEngine(b)
    e.Board.update_board(((0, 1), (2, 0), "base"), "white")
    e.game_states.append(e.Board.copy())
    e.undo_move()
    assert len(e.game_states) == 1
    assert e.Board.equal(Board())
    assert e.Board.turn == "white"


def test_choose_move_computer_mode_applies_move():
    e = GameEngine(Board(), player1="computer", player2="computer")
    e.choose_move(((0, 1), (2, 0), "base"))
    assert isinstance(e.Board.board[2, 0], Knight)
    assert e.move_counter == 1
    assert len(e.game_states) == 2
    assert e.get_turn()[0] == "black"
