import sys
import numpy as np
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.rook import Rook
from darkchess.src.king import King


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].get_position() == (i, j)


def test_board_default_start_positions_and_turn():
    board = Board()
    assert board.turn == "white"
    assert board.board[0, 0].get_symbol() == "wr "
    assert board.board[0, 1].get_symbol() == "wk "
    assert board.board[0, 3].get_symbol() == "wq "
    assert board.board[0, 4].get_symbol() == "wK "
    assert board.board[1, 0].get_symbol() == "wp "
    assert board.board[6, 0].get_symbol() == "bp "
    assert board.board[7, 0].get_symbol() == "br "
    assert board.board[7, 4].get_symbol() == "bK "


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    rendered = str(board)
    assert rendered.startswith("  A  B  C  D  E  F  G  H")
    assert rendered.strip().endswith("A  B  C  D  E  F  G  H")
    assert "wK " in rendered
    assert "bK " in rendered
    assert "1 " in rendered
    assert "8 " in rendered


def test_board_copy_is_deep_and_equal_matches_same_position():
    board = Board()
    copied = board.copy()
    assert board.equal(copied)
    assert board is not copied
    assert board.board is not copied.board
    assert board.board[0, 1] is not copied.board[0, 1]
    copied.board[0, 1].set_position((2, 2))
    assert board.board[0, 1].get_position() == (0, 1)
    assert copied.board[0, 1].get_position() == (2, 2)
    assert board.equal(copied)


def test_board_equal_detects_different_symbol_positions():
    board1 = Board()
    board2 = board1.copy()
    board2.board[0, 1] = Dummy((0, 1))
    assert not board1.equal(board2)


def test_update_board_base_moves_piece_and_toggles_turn():
    board = Board()
    move = ((1, 0), (2, 0), "base")
    board.update_board(move, "white")
    assert board.board[2, 0].get_symbol() == "wp "
    assert isinstance(board.board[1, 0], Dummy)
    assert board.board[2, 0].moved is True
    assert board.turn == "black"


def test_update_board_enpassant_clears_captured_square():
    board = clean_board()
    board[4, 4] = Knight((4, 4), "white")
    board[4, 5] = Knight((4, 5), "black")
    board.turn = "white"
    board.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert board.board[5, 5].get_symbol() == "wk "
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert board.turn == "black"


def test_update_board_castling_kingside_moves_rook_and_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 7] = Rook((0, 7), "white")
    board.turn = "white"
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert board.board[0, 6].get_symbol() == "wK "
    assert board.board[0, 5].get_symbol() == "wr "
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 7], Dummy)
    assert board.turn == "black"


def test_update_board_castling_queenside_moves_rook_and_king():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 0] = Rook((0, 0), "white")
    board.turn = "white"
    board.update_board(((0, 4), (0, 2), "castling"), "white")
    assert board.board[0, 2].get_symbol() == "wK "
    assert board.board[0, 3].get_symbol() == "wr "
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 0], Dummy)


def test_update_board_evolve_promotes_to_queen_and_toggles_turn():
    board = clean_board()
    board[6, 0] = Knight((6, 0), "white")
    board.turn = "white"
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert board.board[7, 0].get_symbol() == "wq "
    assert isinstance(board.board[6, 0], Dummy)
    assert board.turn == "black"


def test_knight_center_moves_include_all_eight_targets_when_empty():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    moves = knight.get_moves([board])
    targets = {m[1] for m in moves}
    expected = {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
    assert targets == expected
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_does_not_move_to_square_occupied_by_same_team():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    board[1, 2] = Knight((1, 2), "white")
    moves = knight.get_moves([board])
    assert (3, 3), (1, 2), "base" not in moves
    assert (3, 3, "base") != ()  # sanity check to keep assertion syntax valid
    assert (3, 3) is not None
    assert (1, 2) not in {m[1] for m in moves}


def test_knight_copy_preserves_state_but_returns_new_object():
    knight = Knight((2, 1), "black")
    knight.moved = True
    copied = knight.copy()
    assert copied is not knight
    assert copied.get_position() == (2, 1)
    assert copied.team == "black"
    assert copied.moved is True
    copied.set_position((4, 4))
    assert knight.get_position() == (2, 1)


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board(), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    moves = {
        1: ((0, 0), (2, 1), "base"),
        2: ((7, 7), (5, 6), "base"),
    }
    translated = GameEngine.num_alph(moves)
    assert translated == {1: ["A1", "B3"], 2: ["H8", "G6"]}


def test_gameengine_compare_states_and_all_moves():
    board = Board()
    engine = GameEngine(board)
    assert engine.compare_states(board, board.copy()) is True
    moves = engine.all_moves()
    assert isinstance(moves, list)
    assert len(moves) > 0
    assert any(move[0] == (1, 0) for move in moves)


def test_is_stalemate_no_move_exits_when_no_legal_moves_and_not_in_check(monkeypatch):
    engine = GameEngine(Board())
    engine.all_moves = lambda: []
    engine.white_check = False
    engine.black_check = False
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_is_checkmate_exits_when_no_escape_moves(monkeypatch):
    engine = GameEngine(Board())
    engine.undo_move = lambda: None
    engine.all_moves = lambda: []
    engine.is_check = lambda: None
    with pytest.raises(SystemExit):
        engine.is_checkmate("white")


def test_is_stalemate_3states_exits_on_repeated_states():
    board = Board()
    engine = GameEngine(board)
    engine.game_states = [board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")
