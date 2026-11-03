import sys
import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.king import King
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert board[i, j].team is None
            assert board[i, j].symbol == "   "


def test_board_default_start_positions_and_turn():
    board = Board()
    assert board.turn == "white"
    assert isinstance(board.board[0, 0], type(Board().board[0, 0]))
    assert board.board[0, 0].get_symbol() == "wr "
    assert board.board[0, 4].get_symbol() == "wK "
    assert board.board[7, 4].get_symbol() == "bK "
    for j in range(8):
        assert board.board[1, j].get_symbol() == "wp "
        assert board.board[6, j].get_symbol() == "bp "


def test_board_string_contains_coordinates_and_piece_symbols():
    board = Board()
    s = str(board)
    assert "A  B  C  D  E  F  G  H" in s
    assert "8 " in s and "1 " in s
    assert "wK " in s
    assert "bK " in s


def test_board_copy_is_deep_and_equal_to_original():
    board = Board()
    copied = board.copy()
    assert board.equal(copied)
    assert copied is not board
    assert copied.board is not board.board
    copied.board[1, 0] = Dummy((1, 0), None)
    assert not board.equal(copied)
    assert board.board[1, 0].get_symbol() == "wp "


def test_board_update_board_base_moves_piece_and_toggles_turn():
    board = Board()
    start = (1, 0)
    end = (2, 0)
    board.update_board((start, end, "base"), "white")
    assert board.board[end].get_symbol() == "wp "
    assert board.board[start].team is None
    assert board.board[end].moved is True
    assert board.turn == "black"


def test_board_update_board_enpassant_clears_captured_pawn():
    board = Board()
    start = (4, 4)
    end = (5, 5)
    board.board[start] = Pawn(start, "white")
    board.board[(4, 5)] = Pawn((4, 5), "black")
    board.update_board((start, end, "enpasse"), "white")
    assert board.board[end].get_symbol() == "wp "
    assert board.board[start].team is None
    assert board.board[(4, 5)].team is None
    assert board.turn == "black"


def test_board_update_board_castling_kingside_moves_rook():
    board = Board()
    board.board = clean_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Knight((0, 7), "white")  # temporary occupant replaced by rook-like movable piece not needed
    board.board[0, 7] = Queen((0, 7), "white")
    board.board[0, 0] = Queen((0, 0), "white")
    board.update_board((0, 4, 4), (0, 6), "castling")
    assert board.board[0, 6].get_symbol() == "wK "
    assert board.board[0, 5].team == "white"
    assert board.board[0, 7].team is None


def test_board_update_board_evolve_promotes_to_queen():
    board = Board()
    start = (6, 0)
    end = (7, 0)
    board.board[start] = Pawn(start, "white")
    board.update_board((start, end, "evolve"), "white")
    assert board.board[end].get_symbol() == "wQ "
    assert board.board[start].team is None
    assert board.turn == "black"


def test_board_update_board_invalid_turn_raises_value_error():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), "base"), "white")


def test_knight_symbol_and_value_and_copy():
    k = Knight((3, 3), "white")
    assert k.get_name() == "knight"
    assert k.get_value() == 3
    assert k.get_symbol() == "wk "
    k.moved = True
    k2 = k.copy()
    assert k2 is not k
    assert k2.get_position() == (3, 3)
    assert k2.team == "white"
    assert k2.moved is True


def test_knight_moves_from_center_has_eight_moves():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    game_states = [board.copy()]
    moves = knight.get_moves(game_states)
    expected = {
        (3, 3, 1, 2), (3, 3, 1, 4), (3, 3, 2, 1), (3, 3, 4, 1),
        (3, 3, 5, 2), (3, 3, 5, 4), (3, 3, 4, 5), (3, 3, 2, 5)
    }
    result = {(a[0], a[1], b[0], b[1]) for a, b, _ in moves}
    assert len(moves) == 8
    assert result == expected


def test_knight_moves_blocked_by_same_team_piece():
    board = Board(clean_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    board.board[5, 4] = Pawn((5, 4), "white")
    moves = knight.get_moves([board.copy()])
    assert (3, 3), (5, 4), "base" not in moves
    assert all(move[1] != (5, 4) for move in moves)


def test_knight_get_moves_invalid_team_raises_value_error():
    knight = Knight((3, 3), "green")
    with pytest.raises(ValueError):
        knight.get_moves([Board().copy()])


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

    move_dict = {1: ((0, 0), (1, 2), "base")}
    assert engine.num_alph(move_dict) == {1: ["A1", "C2"]}


def test_gameengine_is_check_with_knight_attack():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[2, 5] = Knight((2, 5), "black")  # attacks (0,4)
    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_stalemate_no_move_exits(monkeypatch):
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)

    def fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_gameengine_undo_move_reverts_to_previous_state():
    board = Board()
    engine = GameEngine(board)
    original_symbol = engine.Board.board[1, 0].get_symbol()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()
    assert engine.Board.board[1, 0].get_symbol() == original_symbol
    assert engine.Board.board[2, 0].team is None
