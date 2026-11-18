import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen


def make_empty_board():
    return Board(starting_board=clean_board())


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
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
    assert board.board[7, 4].get_symbol() == "bK "
    assert board.board[6, 7].get_symbol() == "bp "


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    rep = str(board)
    assert rep.startswith("  A  B  C  D  E  F  G  H")
    assert rep.strip().endswith("A  B  C  D  E  F  G  H")
    assert "wK " in rep
    assert "bK " in rep
    assert "wr " in rep


def test_board_copy_is_deep_and_equal():
    board = Board()
    copied = board.copy()
    assert board.equal(copied)
    assert board is not copied
    assert board.board is not copied.board
    assert board.board[0, 0] is not copied.board[0, 0]
    copied.board[0, 0].moved = True
    assert board.board[0, 0].moved is False


def test_board_update_base_moves_piece_and_toggles_turn():
    board = make_empty_board()
    board.board[1, 0] = Pawn((1, 0), "white")
    board.update_board(((1, 0), (2, 0), "base"), "white")
    assert isinstance(board.board[2, 0], Pawn)
    assert board.board[2, 0].team == "white"
    assert board.board[2, 0].get_position() == (2, 0)
    assert isinstance(board.board[1, 0], Dummy)
    assert board.turn == "black"


def test_board_update_en_passant_clears_captured_pawn():
    board = make_empty_board()
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 5] = Pawn((4, 5), "black")
    board.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert isinstance(board.board[5, 5], Pawn)
    assert board.board[5, 5].team == "white"
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert board.turn == "black"


def test_board_update_castling_kingside_moves_rook_and_king():
    board = make_empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 7], Dummy)
    assert board.board[0, 6].moved is True
    assert board.board[0, 5].moved is True


def test_board_update_castling_queenside_moves_rook_and_king():
    board = make_empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 0] = Rook((0, 0), "white")
    board.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(board.board[0, 2], King)
    assert isinstance(board.board[0, 3], Rook)
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 0], Dummy)


def test_board_update_evolve_replaces_with_queen():
    board = make_empty_board()
    board.board[6, 1] = Pawn((6, 1), "white")
    board.update_board(((6, 1), (7, 1), "evolve"), "white")
    assert isinstance(board.board[7, 1], Queen)
    assert board.board[7, 1].team == "white"
    assert isinstance(board.board[6, 1], Dummy)
    assert board.turn == "black"


def test_board_equal_detects_difference():
    board1 = Board()
    board2 = Board()
    assert board1.equal(board2)
    board2.board[0, 0] = Dummy((0, 0), None)
    assert not board1.equal(board2)


def test_knight_get_moves_from_center_has_eight_moves():
    board = make_empty_board()
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    moves = knight.get_moves([board])
    assert len(moves) == 8
    assert ((3, 3), (1, 2), "base") in moves
    assert ((3, 3), (5, 4), "base") in moves


def test_knight_get_moves_blocks_same_team_and_allows_captures():
    board = make_empty_board()
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight
    board.board[1, 2] = Pawn((1, 2), "white")  # blocked
    board.board[5, 4] = Pawn((5, 4), "black")  # capturable
    moves = knight.get_moves([board])
    assert ((3, 3), (1, 2), "base") not in moves
    assert ((3, 3), (5, 4), "base") in moves


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board())
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    move_dict = {1: ((0, 1), (2, 2), "base")}
    assert engine.num_alph(move_dict) == {1: ["B1", "C3"]}


def test_gameengine_is_check_detects_knight_attack():
    board = make_empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    board.board[2, 5] = Knight((2, 5), "black")  # attacks (0,4)
    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_is_stalemate_no_move_exits(monkeypatch):
    board = make_empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)
    monkeypatch.setattr(engine, "all_moves", lambda: [])
    monkeypatch.setattr("sys.exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_gameengine_undo_move_restores_previous_state():
    board = Board()
    engine = GameEngine(board)
    original = engine.Board.copy()
    engine.Board.update_board(((1, 0), (2, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()
    assert engine.Board.equal(original)
    assert engine.move_counter == 0
