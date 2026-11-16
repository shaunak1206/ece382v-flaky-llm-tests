import numpy as np

from darkchess.src.board import Board
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.piece import Piece
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.gamengine import GameEngine
import pytest


def empty_board():
    """Board with no pieces, only Dummy placeholders."""
    return Board(clean_board())


def test_clean_board_shape_and_cells_are_dummies():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            cell = board[i, j]
            assert isinstance(cell, Dummy)
            assert cell.get_position() == (i, j)
    # every cell is its own object, not one shared instance
    assert board[0, 0] is not board[0, 1]
    assert board[7, 7] is not board[7, 6]


def test_default_board_has_32_pieces_and_32_empty_cells():
    board = Board()
    flat = list(board.board.flat)
    assert sum(isinstance(c, Dummy) for c in flat) == 32
    assert sum(isinstance(c, Piece) and not isinstance(c, Dummy) for c in flat) == 32


def test_default_board_turn_is_white():
    assert Board().turn == "white"


def test_default_board_piece_placement():
    board = Board()
    assert isinstance(board.board[0, 1], Knight)
    assert board.board[0, 1].team == "white"
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[7, 6].team == "black"
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[1, 0], Pawn)
    assert isinstance(board.board[7, 0], Rook)


def test_equal_identical_boards():
    assert Board().equal(Board())


def test_equal_detects_difference():
    b1 = Board()
    b2 = Board()
    b2.update_board(((1, 0), (2, 0), 'base'), 'white')
    assert not b1.equal(b2)
    assert not b2.equal(b1)


def test_copy_is_independent_of_original():
    original = Board()
    dup = original.copy()
    assert dup is not original
    assert dup.board is not original.board
    assert dup.equal(original)
    assert dup.turn == "white"

    original.update_board(((1, 0), (2, 0), 'base'), 'white')
    assert original.turn == "black"
    assert dup.turn == "white"
    assert isinstance(dup.board[1, 0], Pawn)
    assert isinstance(original.board[1, 0], Dummy)


def test_update_board_base_move():
    board = Board()
    pawn = board.board[1, 4]
    board.update_board(((1, 4), (3, 4), 'base'), 'white')
    assert board.board[3, 4] is pawn
    assert pawn.get_position() == (3, 4)
    assert pawn.moved is True
    assert isinstance(board.board[1, 4], Dummy)
    assert board.turn == "black"


def test_update_board_castling_kingside_moves_rook():
    board = empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")
    board.update_board(((0, 4), (0, 6), 'castling'), 'white')

    assert isinstance(board.board[0, 6], King)
    assert board.board[0, 6].get_position() == (0, 6)
    assert isinstance(board.board[0, 5], Rook)
    assert board.board[0, 5].get_position() == (0, 5)
    assert board.board[0, 5].moved is True
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)


def test_update_board_evolve_promotes_to_queen():
    board = empty_board()
    board.board[6, 0] = Pawn((6, 0), "black")
    board.update_board(((6, 0), (7, 0), 'evolve'), 'black')
    promoted = board.board[7, 0]
    assert isinstance(promoted, Queen)
    assert promoted.team == "black"
    assert isinstance(board.board[6, 0], Dummy)


def test_update_board_enpasse_removes_captured_pawn():
    board = empty_board()
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 5] = Pawn((4, 5), "black")
    board.update_board(((4, 4), (5, 5), 'enpasse'), 'white')

    assert isinstance(board.board[5, 5], Pawn)
    assert board.board[5, 5].team == "white"
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)


def test_update_board_invalid_turn_raises():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), 'base'), 'white')


def test_knight_moves_from_corner_region_default_board():
    board = Board()
    knight = board.board[0, 1]
    moves = knight.get_moves([board])
    assert all(m[0] == (0, 1) and m[2] == "base" for m in moves)
    # (1,3) is blocked by own pawn, so only (2,0) and (2,2) remain
    assert sorted(m[1] for m in moves) == [(2, 0), (2, 2)]


def test_knight_moves_center_empty_board():
    board = empty_board()
    knight = Knight((4, 4), "white")
    board.board[4, 4] = knight
    moves = knight.get_moves([board])
    expected = sorted([(2, 3), (2, 5), (3, 2), (3, 6),
                       (5, 2), (5, 6), (6, 3), (6, 5)])
    assert sorted(m[1] for m in moves) == expected


def test_black_knight_blocked_by_own_piece_but_captures_enemy():
    board = empty_board()
    knight = Knight((4, 4), "black")
    board.board[4, 4] = knight
    board.board[6, 5] = Knight((6, 5), "black")   # own piece: blocked
    board.board[2, 3] = Pawn((2, 3), "white")     # enemy piece: capturable
    moves = knight.get_moves([board])
    ends = [m[1] for m in moves]
    assert (6, 5) not in ends
    assert (2, 3) in ends
    assert len(ends) == 7


def test_knight_symbol_name_value():
    assert Knight((0, 1), "white").get_symbol() == 'wk '
    assert Knight((7, 1), "black").get_symbol() == 'bk '
    assert Knight((0, 1), "white").get_name() == "knight"
    assert Knight((0, 1), "white").get_value() == 3


def test_game_engine_init_defaults():
    board = Board()
    engine = GameEngine(board)
    assert engine.Board is board
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False


def test_game_engine_get_turn_alternates():
    engine = GameEngine(Board(), player1="computer")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    engine.move_counter = 2
    assert engine.get_turn() == ("white", "computer")


def test_num_alph_converts_moves_to_chess_notation():
    move_dict = {1: ((1, 4), (3, 4), 'base'), 2: ((0, 1), (2, 0), 'base')}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ['E2', 'E4'], 2: ['B1', 'A3']}


def test_is_check_detects_knight_attack():
    board = empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[2, 3] = Knight((2, 3), "black")  # attacks (0,4)
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((1, 4), (3, 4), 'base'), 'white')
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert isinstance(engine.Board.board[1, 4], Pawn)
    assert isinstance(engine.Board.board[3, 4], Dummy)


def test_human_move_plays_through_monkeypatched_input(monkeypatch):
    engine = GameEngine(Board())
    answers = iter(["b1", "1"])  # select knight on B1, then option 1 (A3)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    engine.choose_move()

    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[2, 0], Knight)
    assert engine.Board.board[2, 0].team == "white"
    assert isinstance(engine.Board.board[0, 1], Dummy)
    assert engine.Board.turn == "black"


def test_stalemate_no_move_exits_only_when_no_moves_and_not_in_check():
    # default board: white has moves, so no exit
    assert GameEngine(Board()).is_stalemate_no_move("white") is None

    # white has no pieces at all, not in check -> stalemate -> sys.exit
    board = empty_board()
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_stalemate_3states_detects_repetition():
    # fewer than 9 states: no check is done
    engine = GameEngine(Board())
    assert engine.is_stalemate_3states("white") is None

    # 9 identical states: repetition -> sys.exit
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")
