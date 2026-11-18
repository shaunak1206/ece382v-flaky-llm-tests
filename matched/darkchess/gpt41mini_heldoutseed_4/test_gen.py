import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy

def test_clean_board_returns_8x8_array_of_Dummy():
    board_array = clean_board()
    assert isinstance(board_array, np.ndarray)
    assert board_array.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board_array[i, j], Dummy)
            assert board_array[i, j].pos == (i, j)
            # Team should be None for Dummy pieces created by clean_board
            assert getattr(board_array[i, j], 'team', None) is None

def test_board_default_start_position_sets_pieces_correctly():
    board = Board()
    # Check some white pieces initial positions
    assert board.board[0, 0].get_symbol().startswith('w')  # Rook
    assert board.board[0, 1].get_symbol() == 'wk '  # Knight
    assert board.board[1, 0].get_symbol() == 'wp '  # Pawn
    assert board.board[0, 4].get_symbol() == 'wK '  # King
    # Check some black pieces initial positions
    assert board.board[7, 0].get_symbol().startswith('b')  # Rook
    assert board.board[7, 1].get_symbol() == 'bk '  # Knight
    assert board.board[6, 0].get_symbol() == 'bp '  # Pawn
    assert board.board[7, 4].get_symbol() == 'bK '  # King

def test_board_str_includes_rank_and_file_labels_and_pieces():
    board = Board()
    board_str = str(board)
    # Board string should include ranks 1 to 8 and files A to H
    for rank in map(str, range(1, 9)):
        assert rank in board_str
    for file in "ABCDEFGH":
        assert file in board_str
    # Check that some known piece symbol is present as substring
    assert 'wK ' in board_str or 'bK ' in board_str

def test_board_update_board_base_move_changes_positions_and_turn():
    board = Board()
    start_pos = (1, 0)  # a white pawn initial position
    end_pos = (3, 0)    # move two steps forward
    move = (start_pos, end_pos, None)  # base move (treated as 'base')
    old_piece = board.board[start_pos]
    board.update_board(move, "white")
    assert board.board[end_pos].get_symbol() == old_piece.get_symbol()
    assert isinstance(board.board[start_pos], Dummy)
    assert board.turn == "black"
    # The moved piece's pos attribute should update to new position
    assert board.board[end_pos].pos == end_pos
    assert board.board[end_pos].moved is True

def test_board_update_board_enpasse_removes_correct_pawn():
    board = Board(clean_board())
    # Place white pawn at (3,4), black pawn at (3,5) for en passant scenario
    from darkchess.src.pawn import Pawn
    board.board[3,4] = Pawn((3,4), "white")
    board.board[3,5] = Pawn((3,5), "black")
    # Move white pawn from (3,4) to (2,5) en passant capture
    move = ((3,4), (2,5), "enpasse")
    board.turn = "white"
    board.update_board(move, "white")
    # The captured pawn at (3,5) should be Dummy now
    assert isinstance(board.board[(3,5)], Dummy)
    assert board.board[(3,5)].pos == (3,5)
    # The moved pawn should be at (2,5)
    assert board.board[(2,5)].get_symbol().startswith('w')
    # Start position should be Dummy
    assert isinstance(board.board[(3,4)], Dummy)
    assert board.turn == "black"

def test_board_update_board_castling_moves_rook_and_king():
    board = Board(clean_board())
    from darkchess.src.rook import Rook
    from darkchess.src.king import King
    # Set up white king and rook for queenside castling on row 0
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 0] = Rook((0, 0), "white")
    # Clear spaces between king and rook
    for j in range(1, 4):
        board.board[0, j] = Dummy((0, j))
    # Perform castling move: king moves from (0,4) to (0,2)
    move = ((0,4), (0,2), "castling")
    board.turn = "white"
    board.update_board(move, "white")
    # King should be on (0,2)
    assert board.board[(0,2)].get_symbol() == 'wK '
    # Rook should have moved to (0,3)
    assert board.board[(0,3)].get_symbol() == 'wR '
    # Positions (0,0) and (0,4) should be Dummy
    assert isinstance(board.board[(0,0)], Dummy)
    assert isinstance(board.board[(0,4)], Dummy)
    assert board.turn == "black"

def test_board_update_board_evolve_queen_and_clear_start():
    board = Board(clean_board())
    from darkchess.src.pawn import Pawn
    # Place white pawn in position to promote
    board.board[6, 0] = Pawn((6, 0), "white")
    move = ((6,0), (7,0), "evolve")
    board.turn = "white"
    board.update_board(move, "white")
    # The moved piece should be white queen on (7,0)
    assert board.board[(7,0)].get_symbol() == 'wQ '
    # Start position is Dummy
    assert isinstance(board.board[(6,0)], Dummy)
    # Turn switched to black
    assert board.turn == "black"

def test_board_copy_creates_deep_copy_of_board():
    board = Board()
    board_copy = board.copy()
    # The copy should be a different object
    assert board_copy is not board
    # Boards should be equal in positions
    assert board.equal(board_copy)
    # Modify the copy and check the original is unchanged
    start_pos = (1, 0)
    end_pos = (2, 0)
    move = (start_pos, end_pos, None)
    board_copy.update_board(move, "white")
    assert not board.equal(board_copy)
    # Original board at start_pos should remain a white pawn
    assert board.board[start_pos].get_symbol() == 'wp '

def test_board_equal_returns_false_for_different_positions():
    board1 = Board()
    board2 = Board()
    # Modify board2 at one position
    from darkchess.src.knight import Knight
    board2.board[0, 1] = Knight((0, 1), "white")
    # board1 already has knight there, so make a change to board2 to break equality
    board2.board[0, 1] = Knight((0, 1), "black")
    assert board1.equal(board2) is False

def test_knight_get_name_value_and_symbol_for_teams():
    knight_white = Knight((0, 1), "white")
    knight_black = Knight((7, 6), "black")
    assert knight_white.get_name() == "knight"
    assert knight_black.get_name() == "knight"
    assert knight_white.get_value() == 3
    assert knight_black.get_value() == 3
    assert knight_white.get_symbol() == "wk "
    assert knight_black.get_symbol() == "bk "
    with pytest.raises(ValueError):
        Knight((0,0), "green").get_symbol()

def test_knight_get_moves_returns_valid_and_no_invalid_positions():
    board = Board()
    # place a white knight roughly in the center
    knight = Knight((4,4), "white")
    board.board[4,4] = knight
    game_states = [board.copy()]
    moves = knight.get_moves(game_states)
    # Moves should be within board and no friendly fire
    for move in moves:
        start, end, move_type = move
        assert start == (4,4)
        x,y = end
        assert 0 <= x <= 7
        assert 0 <= y <= 7
        # target square must not be occupied by same team
        target_piece = game_states[-1].board[end]
        if target_piece.team is not None:
            assert target_piece.team != "white"
        assert move_type == "base"

def test_knight_copy_creates_independent_knight_with_same_properties():
    knight = Knight((3,3), "black")
    knight.moved = True
    knight_copy = knight.copy()
    assert isinstance(knight_copy, Knight)
    assert knight_copy.team == knight.team
    assert knight_copy.pos == knight.pos
    assert knight_copy.moved == knight.moved
    # changing copy's pos or moved should not affect original
    knight_copy.pos = (4,4)
    knight_copy.moved = False
    assert knight.pos == (3,3)
    assert knight.moved is True

def test_gameengine_get_turn_and_move_counter():
    board = Board()
    engine = GameEngine(board)
    # Initial move counter 0 -> white, human player1
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"
    # Advance a move counter, should be black and human player2
    engine.move_counter = 1
    team, player = engine.get_turn()
    assert team == "black"
    assert player == "human"

def test_gameengine_num_alph_translates_moves_to_readable_format():
    moves = [((1,0),(3,0),"base"), ((0,1),(2,2),"base")]
    move_dict = {1: moves[0], 2: moves[1]}
    expected = {1: ['A2', 'A4'], 2: ['B1', 'C3']}
    result = GameEngine.num_alph(move_dict)
    assert result == expected

def test_gameengine_compare_states_returns_true_for_equal_boards():
    b1 = Board()
    b2 = b1.copy()
    engine = GameEngine(b1)
    assert engine.compare_states(b1, b2) is True
    # Modify b2 and test again
    from darkchess.src.knight import Knight
    b2.board[0,1] = Knight((0,1), "black")
    assert engine.compare_states(b1, b2) is False
