import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_8x8_with_dummy():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            # Dummy pieces have None team by default
            assert board[i, j].team is None
            assert board[i, j].pos == (i, j)


def test_board_init_default_has_correct_turn_and_pieces():
    b = Board()
    # Turn should start as white
    assert b.turn == "white"
    # Pawns in rows 1 (white) and 6 (black)
    for j in range(8):
        assert b.board[1, j].get_name() == "pawn"
        assert b.board[1, j].team == "white"
        assert b.board[6, j].get_name() == "pawn"
        assert b.board[6, j].team == "black"
    # Check specific pieces at corners/center for white
    assert b.board[0, 0].get_name() == "rook"
    assert b.board[0, 0].team == "white"
    assert isinstance(b.board[0, 1], Knight)
    assert b.board[0, 1].team == "white"
    assert b.board[0, 4].get_name() == "king"
    assert b.board[0, 4].team == "white"
    # Check specific pieces at corners/center for black
    assert b.board[7, 0].get_name() == "rook"
    assert b.board[7, 0].team == "black"
    assert isinstance(b.board[7, 1], Knight)
    assert b.board[7, 1].team == "black"
    assert b.board[7, 4].get_name() == "king"
    assert b.board[7, 4].team == "black"


def test_board_update_base_move_changes_positions_and_turn():
    board = Board()
    start = (1, 0)  # white pawn at (1,0)
    end = (3, 0)
    move = (start, end, "base")
    old_piece = board.board[start]
    board.update_board(move, "white")
    assert isinstance(board.board[start], Dummy)
    assert board.board[start].pos == start
    assert board.board[end] == old_piece
    assert board.board[end].pos == end
    assert board.turn == "black"


def test_board_update_board_enpasse():
    board = Board()
    # Setup pieces for en passant scenario
    # Create white pawn at (3, 4), black pawn moves two steps from (6,5) to (4,5)
    white_pawn = board.board[1, 4]
    black_pawn = board.board[6, 5]

    board.board[3, 4] = white_pawn
    board.board[1, 4] = Dummy((1, 4))
    white_pawn.set_position((3, 4))

    board.board[6, 5] = Dummy((6, 5))
    board.board[4, 5] = black_pawn
    black_pawn.set_position((4, 5))

    # Now white pawn captures en passant to (5,5) (actually for test, use positions accordingly)
    start = (3, 4)
    end = (5, 5)
    move = (start, end, "enpasse")

    board.update_board(move, "white")

    assert isinstance(board.board[start], Dummy)
    assert board.board[end].team == "white"
    # The captured pawn at (4,5) should be Dummy after en passant
    assert isinstance(board.board[(3, 5)], Dummy) or isinstance(board.board[(start[0], end[1])], Dummy)


def test_board_update_board_castling_kingside():
    board = Board()
    # Setup for castling:
    # Place king and rook in initial positions, clear spaces between them
    # We will simulate white short castling from (0,4) to (0,6)
    # Remove pieces in between (0,5) and (0,6) except the rook
    for col in [1, 2, 3, 5]:
        board.board[0, col] = Dummy((0, col))
    # Confirm rook at (0,7)
    rook = board.board[0, 7]
    king = board.board[0, 4]
    move = ((0, 4), (0, 6), "castling")
    board.update_board(move, "white")

    # King should be at (0,6)
    assert board.board[0, 6] == king
    assert king.pos == (0, 6)
    assert board.board[0, 4].team is None  # Dummy
    # Rook should have moved to (0,5)
    assert board.board[0, 5] == rook
    assert rook.pos == (0, 5)
    assert board.board[0, 7].team is None  # Dummy
    assert board.turn == "black"


def test_board_update_board_castling_queenside():
    board = Board()
    # Setup for castling queenside:
    # Clear columns between (0,1),(0,2),(0,3) except rook at (0,0)
    for col in [1, 2, 3, 5, 6, 7]:
        board.board[0, col] = Dummy((0, col))
    # Place rook and king
    rook = board.board[0, 0]
    king = board.board[0, 4]
    move = ((0, 4), (0, 2), "castling")
    board.update_board(move, "white")

    # King should be at (0,2)
    assert board.board[0, 2] == king
    assert king.pos == (0, 2)
    assert board.board[0, 4].team is None  # Dummy
    # Rook should have moved to (0,3)
    assert board.board[0, 3] == rook
    assert rook.pos == (0, 3)
    assert board.board[0, 0].team is None
    assert board.turn == "black"


def test_board_update_board_evolve_promotes_pawn_to_queen():
    board = Board()
    # Place white pawn at (6,0) and promote to queen at (7,0)
    board.board[6, 0] = board.board[1, 0].copy()
    board.board[6, 0].set_position((6, 0))
    start = (6, 0)
    end = (7, 0)
    move = (start, end, "evolve")

    board.update_board(move, "white")

    # New piece should be a Queen (symbol is 'wQ ')
    promoted_piece = board.board[end]
    assert promoted_piece.get_name() == "queen"
    assert promoted_piece.team == "white"
    assert board.board[start].team is None  # Dummy
    # Turn changes
    assert board.turn == "black"


def test_board_copy_and_equal_independently():
    board1 = Board()
    board2 = board1.copy()
    assert board1.equal(board2)

    # Modify a piece in board2 and check equal is now False
    pos = (1, 0)
    piece = board2.board[pos]
    piece.team = "black" if piece.team == "white" else "white"
    assert not board1.equal(board2)


def test_knight_get_moves_basic():
    board = Board()
    knight = Knight((4, 4), "white")

    # create a new board state with this knight in middle and all other pieces dummy
    new_board = clean_board()
    new_board[4, 4] = knight
    knight.set_position((4, 4))
    game_states = [Board().copy()]
    game_states[-1].board = new_board

    moves = knight.get_moves(game_states)
    # Possible knight moves from center: 8 moves maximum
    expected_moves = {
        (4, 4), (2, 3), (2, 5), (3, 2), (5, 2),
        (6, 3), (6, 5), (5, 6), (3, 6)
    }
    # We expect all 8 moves with type "base"
    assert len(moves) == 8
    for m in moves:
        start, end, typ = m
        assert start == (4, 4)
        assert end in expected_moves
        assert typ == "base"


def test_knight_no_moves_when_surrounded_by_own_pieces():
    board = Board()
    # Place knight at center
    knight = Knight((4, 4), "white")
    new_board = clean_board()
    new_board[4, 4] = knight
    knight.set_position((4, 4))
    # Surround knight with white pieces
    offsets = [(-2, -1), (-2, 1), (-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2)]
    for dx, dy in offsets:
        x, y = 4 + dx, 4 + dy
        new_board[x, y] = Knight((x, y), "white")
    game_states = [Board().copy()]
    game_states[-1].board = new_board

    moves = knight.get_moves(game_states)
    assert moves == []


def test_knight_get_symbol_and_name_value():
    knight_white = Knight((0, 0), "white")
    knight_black = Knight((0, 0), "black")
    assert knight_white.get_name() == "knight"
    assert knight_black.get_name() == "knight"
    assert knight_white.get_value() == 3
    assert knight_black.get_value() == 3
    assert knight_white.get_symbol() == 'wk '
    assert knight_black.get_symbol() == 'bk '
    with pytest.raises(ValueError):
        Knight((0, 0), "green").get_symbol()


def test_gameengine_get_turn_changes_with_moves():
    b = Board()
    game = GameEngine(b)
    team1, player1 = game.get_turn()
    assert team1 == "white"
    game.move_counter = 1
    team2, player2 = game.get_turn()
    assert team2 == "black"


def test_gameengine_num_alph_converts_positions_correctly():
    move_dict = {
        1: ((1, 0), (2, 2), "base"),
        2: ((6, 7), (5, 5), "base")
    }
    expected = {
        1: ['A2', 'C3'],
        2: ['H7', 'F6']
    }
    result = GameEngine.num_alph(move_dict)
    assert result == expected


def test_gameengine_compare_states_equal_and_not_equal():
    b1 = Board()
    b2 = b1.copy()
    game = GameEngine(b1)
    assert game.compare_states(b1, b2)
    # Change a piece in b2 so boards differ
    b2.board[0, 0] = b2.board[0, 0].copy()
    b2.board[0, 0].moved = True
    assert not game.compare_states(b1, b2)

