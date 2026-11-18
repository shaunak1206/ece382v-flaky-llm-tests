import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.knight import Knight
from darkchess.src.gamengine import GameEngine
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_8x8_with_Dummy():
    board_array = clean_board()
    assert isinstance(board_array, np.ndarray)
    assert board_array.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            piece = board_array[i, j]
            # Dummy pieces have team None
            assert isinstance(piece, Dummy)
            assert piece.pos == (i, j)
            assert piece.team is None


def test_board_default_initialization_sets_correct_pieces():
    board = Board()
    # Test some known positions white
    assert board.board[0, 0].get_name() == "rook"
    assert board.board[0, 1].get_name() == "knight"
    assert board.board[1, 3].get_name() == "pawn"
    # Test some known positions black
    assert board.board[7, 4].get_name() == "king"
    assert board.board[6, 7].get_name() == "pawn"
    # Check that turn starts as white
    assert board.turn == "white"


def test_board_str_contains_rows_and_columns_labels():
    board = Board()
    s = str(board)
    assert "A" in s and "H" in s
    assert "8" in s and "1" in s
    assert len(s.splitlines()) >= 10  # header + 8 rows + footer


def test_board_update_board_base_move_updates_correctly():
    board = Board()
    start = (1, 0)  # white pawn initial position
    end = (2, 0)
    move = (start, end, "base")
    team = "white"
    piece_before = board.board[start]
    assert not piece_before.moved
    board.update_board(move, team)
    piece_after = board.board[end]
    # The piece moved to new location
    assert piece_after.get_name() == piece_before.get_name()
    assert piece_after.pos == end
    # The old position should be Dummy
    assert isinstance(board.board[start], Dummy)
    # Piece should be marked as moved
    assert piece_after.moved
    # Turn should switch
    assert board.turn == "black"


def test_board_update_board_evolve_replaces_with_queen():
    board = Board()
    start = (1, 0)
    end = (0, 0)
    team = "white"
    move = (start, end, "evolve")
    board.update_board(move, team)
    piece = board.board[end]
    # Should now be a queen
    assert piece.get_name() == "queen"
    # Original position is Dummy
    assert isinstance(board.board[start], Dummy)


def test_board_update_board_enpasse_removes_captured_pawn():
    board = Board()
    # Setup initial position in a way en passant can happen:
    # white pawn at (3, 4), black pawn at (3, 5), white moves from (3,4) to (4,5) en passant capture
    board.board[3, 4] = board.board[1, 4].copy()
    board.board[3, 4].set_position((3, 4))
    board.board[3, 5] = board.board[6, 5].copy()
    board.board[3, 5].set_position((3, 5))
    start = (3, 4)
    end = (2, 5)
    move = (start, end, "enpasse")
    team = "white"
    board.update_board(move, team)
    # The start position cleared
    assert isinstance(board.board[start], Dummy)
    # The end position has the piece
    assert board.board[end].get_name() == "pawn"
    # The pawn captured en passant should be removed (at (3,5)'s row and end column)
    assert isinstance(board.board[(3, 5)], Dummy) or board.board[(3, 5)].team != "black"


def test_board_update_board_castling_king_side_moves_rook():
    board = Board()
    # Place king and rook in known positions with no movement
    # Castling king side for white: king moves from (0,4) to (0,6), rook from (0,7) to (0,5)
    start = (0, 4)
    end = (0, 6)
    move = (start, end, "castling")
    team = "white"
    board.update_board(move, team)
    # King moved
    assert board.board[end].get_name() == "king"
    assert board.board[end].pos == end
    # Rook relocated
    rook_pos = (0, 5)
    rook = board.board[rook_pos]
    assert rook.get_name() == "rook"
    assert rook.pos == rook_pos 
    # Old king and rook positions should be Dummy
    assert isinstance(board.board[start], Dummy)
    assert isinstance(board.board[(0, 7)], Dummy)


def test_board_update_board_castling_queen_side_moves_rook():
    board = Board()
    # Castling queen side for white: king (0,4) -> (0,2), rook (0,0) -> (0,3)
    start = (0, 4)
    end = (0, 2)
    move = (start, end, "castling")
    team = "white"
    board.update_board(move, team)
    # King moved correctly
    assert board.board[end].get_name() == "king"
    assert board.board[end].pos == end
    # Rook relocated
    rook_pos = (0, 3)
    rook = board.board[rook_pos]
    assert rook.get_name() == "rook"
    assert rook.pos == rook_pos
    # Old king and rook positions cleared
    assert isinstance(board.board[start], Dummy)
    assert isinstance(board.board[(0, 0)], Dummy)


def test_board_copy_returns_deep_copy():
    board = Board()
    board_copy = board.copy()
    # Different objects in memory
    assert board is not board_copy
    # Board arrays should be independent objects
    assert board.board is not board_copy.board
    # Corresponding pieces are equal but independent instances
    for i in range(8):
        for j in range(8):
            p1 = board.board[i, j]
            p2 = board_copy.board[i, j]
            assert p1.get_symbol() == p2.get_symbol()
            assert p1.pos == p2.pos
            assert p1 is not p2
    # Changing original does not affect copy
    board.board[0, 0] = Dummy((0, 0))
    assert board_copy.board[0, 0].get_symbol() != board.board[0, 0].get_symbol()


def test_board_equal_returns_true_for_identical_boards():
    board1 = Board()
    board2 = Board()
    assert board1.equal(board2) is True


def test_board_equal_returns_false_for_different_boards():
    board1 = Board()
    board2 = Board()
    # Modify one piece
    board2.board[0, 0] = Dummy((0, 0))
    assert board1.equal(board2) is False


def test_knight_get_name_and_value_and_symbol_white_and_black():
    knight_w = Knight((0, 1), "white")
    knight_b = Knight((7, 1), "black")
    assert knight_w.get_name() == "knight"
    assert knight_w.get_value() == 3
    assert knight_w.get_symbol() == 'wk '
    assert knight_b.get_symbol() == 'bk '
    # Invalid team raises ValueError
    knight_invalid = Knight((4, 4), "green")
    with pytest.raises(ValueError):
        knight_invalid.get_symbol()


def test_knight_get_moves_returns_valid_moves():
    # Setup an empty board with dummy pieces except one knight in middle
    board_array = clean_board()
    knight = Knight((4, 4), "white")
    board_array[4, 4] = knight
    board = Board(board_array)
    game_states = [board]
    moves = knight.get_moves(game_states)
    # Knight should have up to 8 moves from center
    expected_positions = {(2, 3), (2, 5), (3, 2), (5, 2), (6, 3), (6, 5), (5, 6), (3, 6)}
    returned_positions = {move[1] for move in moves}
    assert expected_positions == returned_positions
    # Moves are all 'base' type
    assert all(move[2] == "base" for move in moves)


def test_knight_get_moves_blocked_by_team_piece():
    # Knight surrounded by white pieces blocks moves
    board_array = clean_board()
    knight = Knight((4, 4), "white")
    board_array[4, 4] = knight
    # Put white dummy pieces in all knight moves locations, but they count as Dummy so different team; to simulate blocking, place white pawns at some
    from darkchess.src.pawn import Pawn
    blocking = [
        (2, 3), (3, 2), (5, 2)
    ]
    for pos in blocking:
        board_array[pos] = Pawn(pos, "white")
    board = Board(board_array)
    game_states = [board]
    moves = knight.get_moves(game_states)
    # Moves containing blocking squares should not be included
    blocked_positions = set(blocking)
    returned_positions = {move[1] for move in moves}
    for blocked_pos in blocked_positions:
        assert blocked_pos not in returned_positions


def test_gameengine_initialization_defaults():
    board = Board()
    engine = GameEngine(board)
    assert engine.move_counter == 0
    assert engine.Board == board
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.black_check is False
    assert engine.white_check is False
    # game_states initial length
    assert len(engine.game_states) == 1


def test_gameengine_get_turn_and_num_alph_and_compare_states():
    board = Board()
    engine = GameEngine(board, player1="human", player2="computer")
    # First move is white's turn
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"
    # Increment and check black turn
    engine.move_counter = 1
    team2, player2 = engine.get_turn()
    assert team2 == "black"
    assert player2 == "computer"
    # num_alph conversion
    move_dict = {1: ((1, 0), (2, 0), "base")}
    converted = engine.num_alph(move_dict)
    assert converted[1] == ['A2', 'A3']
    # compare_states returns True for identical
    bcopy = board.copy()
    assert engine.compare_states(board, bcopy) is True
    # compare_states returns False for different
    bcopy.board[0, 0] = Dummy((0, 0))
    assert engine.compare_states(board, bcopy) is False
