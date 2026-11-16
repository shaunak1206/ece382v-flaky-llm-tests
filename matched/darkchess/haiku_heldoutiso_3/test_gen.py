import io
import sys
from contextlib import redirect_stdout

import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board


def make_board(pieces):
    """Build a Board from an empty clean_board with the given (pos, piece) pairs."""
    arr = clean_board()
    for pos, piece in pieces:
        arr[pos] = piece
    return Board(arr)


# ---------- utils.clean_board ----------

def test_clean_board_shape_and_dtype_of_elements():
    arr = clean_board()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (8, 8)


def test_clean_board_all_cells_are_dummy_with_no_team():
    arr = clean_board()
    for i in range(8):
        for j in range(8):
            assert isinstance(arr[i, j], Dummy)
            assert arr[i, j].team is None


def test_clean_board_dummy_positions_match_indices():
    arr = clean_board()
    for i in range(8):
        for j in range(8):
            assert arr[i, j].get_position() == (i, j)


def test_clean_board_returns_fresh_array_each_call():
    a = clean_board()
    b = clean_board()
    assert a is not b
    assert a[0, 0] is not b[0, 0]


# ---------- Knight ----------

def test_knight_name_and_value():
    k = Knight((0, 0), "white")
    assert k.get_name() == "knight"
    assert k.get_value() == 3


def test_knight_symbol_for_each_team():
    assert Knight((0, 0), "white").get_symbol() == "wk "
    assert Knight((0, 0), "black").get_symbol() == "bk "


def test_knight_symbol_invalid_team_raises():
    with pytest.raises(ValueError):
        Knight((0, 0), "green").get_symbol()


def test_knight_moves_from_center_on_empty_board_gives_eight():
    board = make_board([((3, 3), Knight((3, 3), "white"))])
    moves = board.board[3, 3].get_moves([board])
    ends = {m[1] for m in moves}
    expected = {(1, 2), (1, 4), (2, 1), (2, 5),
                (4, 1), (4, 5), (5, 2), (5, 4)}
    assert len(moves) == 8
    assert ends == expected
    for m in moves:
        assert m[0] == (3, 3)
        assert m[2] == "base"


def test_knight_moves_from_corner_gives_two():
    board = make_board([((0, 0), Knight((0, 0), "white"))])
    moves = board.board[0, 0].get_moves([board])
    assert {m[1] for m in moves} == {(1, 2), (2, 1)}


def test_knight_cannot_move_onto_own_team_piece():
    board = make_board([
        ((3, 3), Knight((3, 3), "white")),
        ((1, 2), Knight((1, 2), "white")),
    ])
    moves = board.board[3, 3].get_moves([board])
    ends = {m[1] for m in moves}
    assert (1, 2) not in ends
    assert len(moves) == 7


def test_knight_can_capture_enemy_piece():
    board = make_board([
        ((3, 3), Knight((3, 3), "white")),
        ((1, 2), Knight((1, 2), "black")),
    ])
    moves = board.board[3, 3].get_moves([board])
    assert (1, 2) in {m[1] for m in moves}
    assert len(moves) == 8


def test_knight_get_moves_invalid_team_raises():
    board = make_board([((3, 3), Knight((3, 3), "green"))])
    with pytest.raises(ValueError):
        board.board[3, 3].get_moves([board])


def test_knight_copy_preserves_state_and_is_new_object():
    k = Knight((2, 5), "black")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "black"
    assert c.get_position() == (2, 5)
    assert c.moved is True


# ---------- Board ----------

def test_board_default_turn_is_white():
    assert Board().turn == "white"


def test_board_default_start_has_32_pieces_split_by_team():
    b = Board()
    whites = blacks = 0
    for i in range(8):
        for j in range(8):
            team = b.board[i, j].team
            if team == "white":
                whites += 1
            elif team == "black":
                blacks += 1
    assert whites == 16
    assert blacks == 16


def test_board_default_start_kings_on_e_file():
    b = Board()
    assert b.board[0, 4].get_symbol() == "wK "
    assert b.board[7, 4].get_symbol() == "bK "


def test_board_uses_given_starting_board_array():
    arr = clean_board()
    b = Board(arr)
    assert b.board is arr


def test_board_update_base_move_moves_piece_and_flips_turn():
    b = make_board([((0, 1), Knight((0, 1), "white"))])
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert b.board[2, 2].team == "white"
    assert b.board[2, 2].get_position() == (2, 2)
    assert b.board[2, 2].moved is True
    assert b.board[0, 1].team is None
    assert b.turn == "black"


def test_board_update_none_keyword_treated_as_base():
    b = make_board([((0, 1), Knight((0, 1), "white"))])
    b.update_board(((0, 1), (2, 2), None), "white")
    assert b.board[2, 2].team == "white"


def test_board_turn_toggles_back_to_white():
    b = make_board([((0, 1), Knight((0, 1), "white")),
                    ((7, 1), Knight((7, 1), "black"))])
    b.update_board(((0, 1), (2, 2), "base"), "white")
    b.update_board(((7, 1), (5, 2), "base"), "black")
    assert b.turn == "white"


def test_board_update_invalid_turn_raises():
    b = make_board([((0, 1), Knight((0, 1), "white"))])
    b.turn = "green"
    with pytest.raises(ValueError):
        b.update_board(((0, 1), (2, 2), "base"), "white")


def test_board_copy_is_independent():
    original = Board()
    dup = original.copy()
    assert dup is not original
    assert dup.turn == original.turn
    assert dup.board[1, 0] is not original.board[1, 0]
    dup.board[1, 0] = Dummy((1, 0), None)
    assert original.board[1, 0].team == "white"


def test_board_equal_identical_and_after_move():
    a = Board()
    b = Board()
    assert a.equal(b)
    b.update_board(((1, 4), (3, 4), "base"), "white")
    assert not a.equal(b)


def test_board_str_has_header_footer_and_eight_rows(): 
    rep = str(Board())
    lines = rep.splitlines()
    assert lines[0] == "  A  B  C  D  E  F  G  H "
    assert lines[-1] == "  A  B  C  D  E  F  G  H "
    assert len(lines) == 10


def test_board_view_board_debug_prints_output(capsys):
    Board().view_board_debug()
    out = capsys.readouterr().out
    assert len(out) > 0


# ---------- GameEngine ----------

def test_game_engine_defaults_and_initial_turn():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")


def test_game_engine_get_turn_black_after_one_move():
    engine = GameEngine(Board(), "human", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")


def test_game_engine_num_alph_converts_coordinates():
    result = GameEngine.num_alph({1: ((1, 4), (3, 4), "base")})
    assert result == {1: ["E2", "E4"]}


def test_game_engine_is_check_false_on_starting_board():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False


def test_game_engine_computer_mode_without_move_raises():
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)


def test_game_engine_computer_mode_plays_legal_move():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert engine.Board.turn == "black"


def test_game_engine_human_mode_rejects_preset_move():
    engine = GameEngine(Board())
    with pytest.raises(ValueError):
        engine.choose_move(((1, 4), (3, 4), "base"))


def test_game_engine_unknown_player_mode_raises():
    engine = GameEngine(Board(), "alien", "human")
    with pytest.raises(ValueError):
        engine.choose_move()


def test_game_engine_undo_move_restores_previous_state():
    engine = GameEngine(Board(), "computer", "computer")
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(Board())


def test_game_engine_is_stalemate_no_move_exits_when_no_moves():
    board = make_board([((3, 3), Knight((3, 3), "black"))])
    engine = GameEngine(board)
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_game_engine_is_stalemate_3states_exits_on_repetition():
    engine = GameEngine(Board())
    engine.game_states = [Board().copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_game_engine_is_stalemate_3states_no_exit_with_few_states():
    engine = GameEngine(Board())
    engine.game_states = [Board().copy() for _ in range(8)]
    assert engine.is_stalemate_3states("white") is None
