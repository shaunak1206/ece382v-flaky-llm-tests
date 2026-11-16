import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.king import King
from darkchess.src.queen import Queen


class _State(object):
    """Minimal stand-in for a game state exposing a `.board` numpy array."""

    def __init__(self, board):
        self.board = board


def _empty_state_with(pieces):
    board = clean_board()
    for (pos, piece) in pieces:
        board[pos] = piece
    return _State(board)


# ---------------------------------------------------------------- Knight

def test_knight_name_and_value():
    knight = Knight((0, 1), "white")
    assert knight.get_name() == "knight"
    assert knight.get_value() == 3


def test_knight_symbol_for_each_team():
    assert Knight((0, 1), "white").get_symbol() == "wk "
    assert Knight((7, 1), "black").get_symbol() == "bk "


def test_knight_symbol_invalid_team_raises():
    with pytest.raises(ValueError):
        Knight((0, 0), "green").get_symbol()


def test_knight_moves_from_center_on_empty_board_has_eight_moves():
    state = _empty_state_with([((4, 4), Knight((4, 4), "white"))])
    moves = Knight((4, 4), "white").get_moves([state])
    targets = {m[1] for m in moves}
    assert len(moves) == 8
    assert targets == {(2, 3), (2, 5), (3, 2), (5, 2),
                       (6, 3), (6, 5), (5, 6), (3, 6)}
    assert all(m[0] == (4, 4) and m[2] == "base" for m in moves)


def test_knight_moves_from_corner_has_two_moves():
    state = _empty_state_with([((0, 0), Knight((0, 0), "white"))])
    moves = Knight((0, 0), "white").get_moves([state])
    assert {m[1] for m in moves} == {(1, 2), (2, 1)}


def test_knight_cannot_land_on_own_piece():
    state = _empty_state_with([
        ((4, 4), Knight((4, 4), "white")),
        ((6, 5), Pawn((6, 5), "white")),
    ])
    moves = Knight((4, 4), "white").get_moves([state])
    targets = {m[1] for m in moves}
    assert (6, 5) not in targets
    assert len(moves) == 7


def test_knight_can_capture_enemy_piece():
    state = _empty_state_with([
        ((4, 4), Knight((4, 4), "white")),
        ((6, 5), Pawn((6, 5), "black")),
    ])
    moves = Knight((4, 4), "white").get_moves([state])
    assert (6, 5) in {m[1] for m in moves}
    assert len(moves) == 8


def test_black_knight_moves_from_center_has_eight_moves():
    state = _empty_state_with([((4, 4), Knight((4, 4), "black"))])
    moves = Knight((4, 4), "black").get_moves([state])
    assert len(moves) == 8


def test_knight_copy_preserves_state_and_is_new_object():
    original = Knight((3, 3), "white")
    original.moved = True
    dup = original.copy()
    assert dup is not original
    assert isinstance(dup, Knight)
    assert dup.team == "white"
    assert dup.moved is True
    assert tuple(dup.get_position()) == (3, 3)


# ---------------------------------------------------------------- utils

def test_clean_board_is_8x8_of_empty_dummies():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].team is None


# ---------------------------------------------------------------- Board

def test_board_default_layout_and_turn():
    board = Board()
    assert board.turn == "white"
    assert isinstance(board.board, np.ndarray)
    assert board.board.shape == (8, 8)
    assert isinstance(board.board[0, 1], Knight)
    assert board.board[0, 1].team == "white"
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[7, 6].team == "black"
    assert isinstance(board.board[1, 3], Pawn)
    assert isinstance(board.board[4, 4], Dummy)


def test_board_accepts_custom_starting_board():
    custom = clean_board()
    custom[0, 4] = King((0, 4), "white")
    board = Board(custom)
    assert board.board is custom
    assert board.turn == "white"


def test_board_update_board_base_move_switches_turn():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    assert isinstance(board.board[3, 4], Pawn)
    assert tuple(board.board[3, 4].get_position()) == (3, 4)
    assert board.board[3, 4].moved is True
    assert isinstance(board.board[1, 4], Dummy)
    assert board.turn == "black"


def test_board_update_board_castling_kingside_moves_rook():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Pawn((0, 7), "white")  # rook stand-in via generic piece
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Pawn)
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 7], Dummy)


def test_board_update_board_enpasse_removes_captured_pawn():
    board = Board(clean_board())
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 3] = Pawn((4, 3), "black")
    board.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(board.board[5, 3], Pawn)
    assert board.board[5, 3].team == "white"
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 3], Dummy)


def test_board_update_board_evolve_promotes_to_queen():
    board = Board(clean_board())
    board.board[6, 0] = Pawn((6, 0), "white")
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert isinstance(board.board[6, 0], Dummy)


def test_board_update_board_invalid_turn_raises():
    board = Board(clean_board())
    board.board[1, 0] = Pawn((1, 0), "white")
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), "base"), "white")


def test_board_copy_is_independent_of_original():
    original = Board()
    dup = original.copy()
    assert dup is not original
    assert dup.turn == original.turn
    dup.board[1, 0] = Dummy((1, 0))
    dup.turn = "black"
    assert isinstance(original.board[1, 0], Pawn)
    assert original.turn == "white"


def test_board_equal_detects_identical_and_different_boards():
    a = Board()
    b = Board()
    assert a.equal(b) is True
    b.board[1, 0] = Dummy((1, 0))
    assert a.equal(b) is False


def test_board_str_has_header_rows_and_footer():
    lines = str(Board()).splitlines()
    assert len(lines) == 10
    assert lines[0] == "  A  B  C  D  E  F  G  H "
    assert lines[-1] == "  A  B  C  D  E  F  G  H "
    assert lines[1].startswith("8 ")
    assert lines[8].startswith("1 ")


def test_board_view_board_prints_string_representation(capsys):
    board = Board()
    board.view_board()
    captured = capsys.readouterr()
    assert captured.out == str(board) + "\n"


# ---------------------------------------------------------------- GameEngine

def test_engine_init_defaults():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False


def test_engine_get_turn_alternates_with_move_counter():
    engine = GameEngine(Board(), "computer", "human")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_engine_num_alph_converts_coordinates():
    move_dict = {1: ((1, 4), (3, 4), "base"), 2: ((0, 1), (2, 2), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["E2", "E4"], 2: ["B1", "C3"]}


def test_engine_is_check_false_in_starting_position():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False


def test_engine_is_check_detects_knight_attack_on_white_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[2, 5] = Knight((2, 5), "black")  # attacks (0, 4)
    board[7, 4] = King((7, 4), "black")
    engine = GameEngine(Board(board))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_engine_all_moves_at_start_is_twenty():
    engine = GameEngine(Board())
    assert len(engine.all_moves()) == 20


def test_engine_choose_move_computer_plays_move():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert engine.Board.board[3, 4].team == "white"
    assert engine.move_counter == 1
    assert engine.get_turn()[0] == "black"


def test_engine_choose_move_computer_without_move_raises():
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move()


def test_engine_choose_move_human_with_move_raises():
    engine = GameEngine(Board())
    with pytest.raises(ValueError):
        engine.choose_move(((1, 4), (3, 4), "base"))


def test_engine_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((1, 4), (3, 4), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(Board()) is True


def test_engine_stalemate_3states_exits_on_repeated_states():
    engine = GameEngine(Board())
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_engine_stalemate_3states_no_exit_with_few_states():
    engine = GameEngine(Board())
    assert engine.is_stalemate_3states("white") is None
