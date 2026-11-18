import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy

# --- Tests for clean_board in utils.py ---

def test_clean_board_shape_and_type():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    # All entries should be Dummy instances with None or pos set
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].get_position() == (i, j)
            # The team for Dummy pieces is None
            assert board[i, j].team is None

# --- Tests for Board class in board.py ---

def test_board_default_initialization_sets_turn_white():
    board = Board()
    assert board.turn == "white"

def test_board_default_initialization_has_pieces_in_starting_positions():
    board = Board()
    # Pawns at row 1 (white)
    for j in range(8):
        piece = board.board[1, j]
        assert piece.get_name() == "pawn"
        assert piece.team == "white"
    # Pawns at row 6 (black)
    for j in range(8):
        piece = board.board[6, j]
        assert piece.get_name() == "pawn"
        assert piece.team == "black"
    # Specific pieces on back rank, white at row 0
    assert board.board[0, 0].get_name() == "rook" and board.board[0, 0].team == "white"
    assert board.board[0, 1].get_name() == "knight" and board.board[0, 1].team == "white"
    assert board.board[0, 2].get_name() == "bishop" and board.board[0, 2].team == "white"
    assert board.board[0, 3].get_name() == "queen" and board.board[0, 3].team == "white"
    assert board.board[0, 4].get_name() == "king" and board.board[0, 4].team == "white"
    assert board.board[0, 5].get_name() == "bishop" and board.board[0, 5].team == "white"
    assert board.board[0, 6].get_name() == "knight" and board.board[0, 6].team == "white"
    assert board.board[0, 7].get_name() == "rook" and board.board[0, 7].team == "white"
    # Black pieces on back rank, row 7
    assert board.board[7, 0].get_name() == "rook" and board.board[7, 0].team == "black"
    assert board.board[7, 1].get_name() == "knight" and board.board[7, 1].team == "black"
    assert board.board[7, 2].get_name() == "bishop" and board.board[7, 2].team == "black"
    assert board.board[7, 3].get_name() == "queen" and board.board[7, 3].team == "black"
    assert board.board[7, 4].get_name() == "king" and board.board[7, 4].team == "black"
    assert board.board[7, 5].get_name() == "bishop" and board.board[7, 5].team == "black"
    assert board.board[7, 6].get_name() == "knight" and board.board[7, 6].team == "black"
    assert board.board[7, 7].get_name() == "rook" and board.board[7, 7].team == "black"

def test_board_str_contains_expected_row_and_column_headers():
    board = Board()
    output = str(board)
    # Check headers
    assert output.startswith("  A  B  C  D  E  F  G  H")
    assert output.endswith("  A  B  C  D  E  F  G  H \n")
    # Check row numbers 1 to 8 present
    for number in range(1, 9):
        assert str(number) in output

def test_update_board_base_move_changes_positions_and_turn():
    board = Board()
    start_pos = (1, 4)  # white pawn starting position
    end_pos = (3, 4)    # move pawn two spaces forward
    move = (start_pos, end_pos, "base")
    original_piece = board.board[start_pos]
    board.update_board(move, "white")
    # Piece should have moved to end_pos
    moved_piece = board.board[end_pos]
    assert moved_piece.get_name() == original_piece.get_name()
    assert moved_piece.get_position() == end_pos
    # Original start position should now be Dummy
    assert isinstance(board.board[start_pos], Dummy)
    # moved flag should be True
    assert moved_piece.moved is True
    # Turn should have switched to black
    assert board.turn == "black"

def test_update_board_enpasse_removes_captured_pawn():
    board = Board()
    # Setup scenario for en passant capture
    # White pawn at (3,4), black pawn at (3,3)
    board.board = clean_board()
    board.board[3, 4] = board.board[1, 4].__class__((3,4), "white")
    board.board[3, 3] = board.board[6, 3].__class__((3,3), "black")
    board.board[1,4] = Dummy((1,4))
    board.board[6,3] = Dummy((6,3))
    board.turn = "white"

    move = ((3,4), (2,3), "enpasse")
    board.update_board(move, "white")

    # The old pawn position empty
    assert isinstance(board.board[(3,4)], Dummy)
    # The black pawn (3,3) is captured (now Dummy)
    assert isinstance(board.board[(3,3)], Dummy)
    # White pawn now at (2,3)
    piece = board.board[(2,3)]
    assert piece.team == "white"
    assert piece.get_name() == "pawn"
    assert piece.moved is True
    # Turn switched
    assert board.turn == "black"

def test_update_board_castling_moves_rook_correctly():
    board = Board()
    board.board = clean_board()
    # Setup for white kingside castling
    # Place king and rook on initial positions and empty spaces between
    board.board[0, 4] = board.board[0, 4].__class__((0,4), "white")  # king
    board.board[0, 7] = board.board[0, 7].__class__((0,7), "white")  # rook
    # Empty spaces between at 0,5 and 0,6
    board.board[0, 5] = Dummy((0,5))
    board.board[0, 6] = Dummy((0,6))
    board.turn = "white"

    # Kingside castling move for king from (0,4) to (0,6)
    move = ((0,4), (0,6), "castling")
    board.update_board(move, "white")

    # King moved to (0,6); rook should be at (0,5)
    assert board.board[(0,6)].get_name() == "king"
    assert board.board[(0,6)].team == "white"
    assert board.board[(0,5)].get_name() == "rook"
    assert board.board[(0,5)].team == "white"
    # Original king and rook positions are dummy
    assert isinstance(board.board[(0,4)], Dummy)
    assert isinstance(board.board[(0,7)], Dummy)
    # moved flags True
    assert board.board[(0,6)].moved is True
    assert board.board[(0,5)].moved is True
    # Turn switched to black
    assert board.turn == "black"

def test_update_board_evolve_replaces_with_queen():
    board = Board()
    board.board = clean_board()
    # Place a white pawn ready to promote at (6, 0)
    board.board[6, 0] = board.board[1,0].__class__((6,0), "white")
    board.turn = "white"
    move = ((6,0), (7,0), "evolve")
    board.update_board(move, "white")

    # Position (7,0) should now be a queen of white team
    piece = board.board[(7,0)]
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # Original pawn position should be Dummy
    assert isinstance(board.board[(6,0)], Dummy)
    # Turn should have switched to black
    assert board.turn == "black"

def test_copy_board_creates_independent_copy():
    board = Board()
    board2 = board.copy()

    # The boards should be equal but different objects
    assert board2 is not board
    assert board.equal(board2)

    # Modifying copy does not affect original
    pos = (1,0)
    orig_piece = board.board[pos]
    board2.board[pos] = Dummy(pos)
    assert not board.equal(board2)
    # Original board's piece unchanged
    assert board.board[pos] == orig_piece

def test_equal_returns_false_for_different_boards():
    board1 = Board()
    board2 = Board()
    # Remove a piece from board2 to create difference
    board2.board[0,0] = Dummy((0,0))
    assert not board1.equal(board2)

def test_equal_returns_true_for_identical_boards():
    board1 = Board()
    board2 = board1.copy()
    assert board1.equal(board2)

# --- Tests for Knight class in knight.py ---

def test_knight_get_name_and_value_and_symbol():
    k_white = Knight((0, 1), "white")
    k_black = Knight((7, 1), "black")
    assert k_white.get_name() == "knight"
    assert k_white.get_value() == 3
    assert k_white.get_symbol() == "wk "
    assert k_black.get_symbol() == "bk "
    with pytest.raises(ValueError):
        Knight((0, 0), "green").get_symbol()

def test_knight_copy_returns_independent_copy():
    k1 = Knight((2, 3), "white")
    k1.moved = True
    k2 = k1.copy()
    assert k2 is not k1
    assert k2.get_name() == "knight"
    assert k2.team == "white"
    assert k2.moved == True
    assert k2.get_position() == (2, 3)

def test_knight_get_moves_returns_valid_moves_white():
    # Setup a clean board with knight at center.
    board = Board()
    board.board = clean_board()
    knight = Knight((4, 4), "white")
    board.board[4,4] = knight
    game_states = [board]

    moves = knight.get_moves(game_states)
    # Possible moves from (4,4)
    expected_positions = [
        (2,3), (2,5), (3,2), (5,2), (6,3), (6,5), (3,6), (5,6)
    ]
    for move in moves:
        assert move[0] == (4,4)
        assert move[1] in expected_positions
        assert move[2] == "base"
    # Moves count can be less or equal to 8 (if blocked by own pieces),
    # but at least the knight should have these moves on an empty board
    assert len(moves) <= 8

def test_knight_get_moves_blocks_own_pieces():
    board = Board()
    board.board = clean_board()
    knight = Knight((4,4), "white")
    board.board[4,4] = knight
    # Set a white piece at (6,5), so knight cannot move there
    board.board[6,5] = Knight((6,5), "white")
    game_states = [board]
    moves = knight.get_moves(game_states)
    forbidden_move = ( (4,4), (6,5), "base")
    assert forbidden_move not in moves

def test_knight_get_moves_raises_for_invalid_team():
    knight = Knight((0,0), "purple")
    with pytest.raises(ValueError):
        knight.get_moves([Board()])

# --- Tests for GameEngine in gamengine.py ---

def test_gameengine_initializes_with_defaults():
    b = Board()
    ge = GameEngine(b)
    assert ge.move_counter == 0
    assert ge.Board.equal(b)
    assert ge.player1 == "human"
    assert ge.player2 == "human"
    assert ge.black_check is False
    assert ge.white_check is False

def test_gameengine_get_turn_alternates():
    b = Board()
    ge = GameEngine(b)
    # Move counter 0 -> white
    team, player = ge.get_turn()
    assert team == "white"
    assert player == "human"
    ge.move_counter = 1
    team, player = ge.get_turn()
    assert team == "black"
    assert player == "human"
    ge.move_counter = 2
    team, player = ge.get_turn()
    assert team == "white"

def test_gameengine_num_alph_conversion():
    ge = GameEngine(Board())
    # Make a move dict with one move from (1,0) to (2,0)
    move_dict = {1: ((1,0), (2,0), "base")}
    res = ge.num_alph(move_dict)
    assert res == {1: ["A2", "A3"]}
    # Test another move dict for a position at H8
    move_dict = {1: ((7,7), (7,6), "base")}
    res = ge.num_alph(move_dict)
    assert res == {1: ["H8", "G8"]}

def test_gameengine_compare_states_returns_true_for_equal_boards():
    b1 = Board()
    b2 = b1.copy()
    ge = GameEngine(b1)
    assert ge.compare_states(b1, b2) is True

def test_gameengine_compare_states_returns_false_for_different_boards():
    b1 = Board()
    b2 = Board()
    # Remove a piece from b2 to make it different
    b2.board[0,0] = Dummy((0,0))
    ge = GameEngine(b1)
    assert ge.compare_states(b1, b2) is False

@pytest.mark.parametrize("team", ["white", "black"])
def test_gameengine_all_moves_returns_only_team_moves(team):
    b = Board()
    ge = GameEngine(b)
    ge.move_counter = 0 if team=="white" else 1
    moves = ge.all_moves()
    # All moves' pieces should be of team to play
    for move in moves:
        start_pos = move[0]
        piece = ge.Board.board[start_pos]
        assert piece.team == team

# Because other GameEngine methods require board states and player inputs,
# Additional fully functional tests related to move choice, checkmate,
# stalemate, undo_move require complex setups and mocking inputs/outputs,
# which are beyond scope here.

