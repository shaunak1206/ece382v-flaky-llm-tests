import pytest
import numpy as np

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen


def test_clean_board_creates_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].pos == (i, j)
            assert board[i, j].team is None


def test_board_default_start_positions():
    board = Board()
    assert isinstance(board.board[0, 0], Rook)
    assert board.board[0, 0].team == "white"
    assert isinstance(board.board[0, 4], King)
    assert board.board[0, 4].team == "white"
    assert isinstance(board.board[7, 3], Queen)
    assert board.board[7, 3].team == "black"
    for j in range(8):
        assert isinstance(board.board[1, j], Pawn)
        assert board.board[1, j].team == "white"
        assert isinstance(board.board[6, j], Pawn)
        assert board.board[6, j].team == "black"


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    board_text = str(board)
    assert "A  B  C  D  E  F  G  H" in board_text
    assert "1 " in board_text
    assert "8 " in board_text
    assert "wk " in board_text
    assert "bk " in board_text


def test_board_update_board_base_moves_piece_and_flips_turn():
    board = Board()
    start = (1, 0)
    end = (2, 0)
    moved_piece = board.board[start]
    board.update_board((start, end, "base"), "white")

    assert isinstance(board.board[start], Dummy)
    assert board.board[start].pos == start
    assert board.board[end].symbol == moved_piece.symbol
    assert board.board[end].pos == end
    assert board.board[end].moved is True
    assert board.turn == "black"


def test_board_update_board_enpassant_clears_captured_square():
    board = clean_board()
    white_pawn = Pawn((4, 4), "white")
    black_pawn = Pawn((4, 5), "black")
    board[4, 4] = white_pawn
    board[4, 5] = black_pawn
    b = Board(board)
    b.turn = "white"

    b.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 5], Dummy)
    assert b.board[5, 5].team == "white"
    assert b.board[5, 5].pos == (5, 5)
    assert b.turn == "black"


def test_board_update_board_castling_moves_rook_correctly_king_side():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 7] = Rook((0, 7), "white")
    b = Board(board)
    b.turn = "white"

    b.update_board(((0, 4), (0, 6), "castling"), "white")

    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)
    assert b.board[0, 6].get_symbol() == "wK "
    assert b.board[0, 6].pos == (0, 6)
    assert b.board[0, 5].get_symbol() == "wr "
    assert b.board[0, 5].pos == (0, 5)
    assert b.turn == "black"


def test_board_update_board_castling_moves_rook_correctly_queen_side():
    board = clean_board()
    board[0, 4] = King((0, 4), "white")
    board[0, 0] = Rook((0, 0), "white")
    b = Board(board)
    b.turn = "white"

    b.update_board(((0, 4), (0, 2), "castling"), "white")

    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 0], Dummy)
    assert b.board[0, 2].get_symbol() == "wK "
    assert b.board[0, 3].get_symbol() == "wr "
    assert b.turn == "black"


def test_board_update_board_evolve_replaces_pawn_with_queen():
    board = clean_board()
    board[6, 0] = Pawn((6, 0), "white")
    b = Board(board)
    b.turn = "white"

    b.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(b.board[6, 0], Dummy)
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert b.turn == "black"


def test_board_copy_is_deep_and_equal_compares_symbols():
    board = Board()
    copied = board.copy()

    assert board.equal(copied) is True
    copied.board[1, 0].moved = True
    copied.board[1, 0].set_position((3, 3))
    assert board.board[1, 0].pos == (1, 0)
    assert board.equal(copied) is True

    copied.board[1, 0] = Dummy((1, 0))
    assert board.equal(copied) is False


def test_knight_get_moves_white_from_center_includes_all_legal_moves():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    game_states = [Board(board)]

    moves = knight.get_moves(game_states)
    destinations = {move[1] for move in moves}
    expected = {(1, 2), (1, 4), (2, 1), (2, 5), (4, 1), (4, 5), (5, 2), (5, 4)}
    assert destinations == expected
    assert all(move[2] == "base" for move in moves)


def test_knight_get_moves_blocks_same_team_and_allows_captures():
    board = clean_board()
    knight = Knight((3, 3), "white")
    board[3, 3] = knight
    board[1, 2] = Pawn((1, 2), "white")
    board[1, 4] = Pawn((1, 4), "black")
    game_states = [Board(board)]

    moves = knight.get_moves(game_states)
    destinations = {move[1] for move in moves}
    assert (1, 2) not in destinations
    assert (1, 4) in destinations


def test_knight_copy_preserves_state_without_sharing_object():
    knight = Knight((2, 1), "black")
    knight.moved = True
    copied = knight.copy()

    assert copied is not knight
    assert copied.pos == knight.pos
    assert copied.team == knight.team
    assert copied.moved is True
    copied.set_position((4, 4))
    assert knight.pos == (2, 1)


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

    move_dict = {1: ((0, 0), (1, 2), "base"), 2: ((7, 7), (5, 6), "base")}
    converted = GameEngine.num_alph(move_dict)
    assert converted == {1: ["A1", "C2"], 2: ["H8", "G6"]}


def test_gameengine_all_moves_uses_current_team(monkeypatch):
    board = Board()
    engine = GameEngine(board)

    def fake_get_moves(pos):
        return [(pos, (pos[0] + 1, pos[1]), "base")]

    monkeypatch.setattr(engine, "get_moves", fake_get_moves)
    moves = engine.all_moves()
    assert len(moves) == 16  # 8 pawns + 8 back-rank pieces in white's starting position


def test_gameengine_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    before = engine.Board.copy()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())

    engine.undo_move()

    assert engine.Board.equal(before) is True
    assert len(engine.game_states) == 1


def test_gameengine_stalemate_no_move_exits(monkeypatch):
    board = clean_board()
    board[0, 0] = King((0, 0), "white")
    board[7, 7] = King((7, 7), "black")
    engine = GameEngine(Board(board))
    engine.white_check = False
    engine.black_check = False
    monkeypatch.setattr(engine, "all_moves", lambda: [])

    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")
