import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.piece import Piece
from darkchess.src.dummy_piece import Dummy

def test_clean_board_returns_8x8_array_of_dummy():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            piece = board[i, j]
            assert isinstance(piece, Dummy)
            assert piece.pos == (i, j)
            assert piece.team is None

def test_board_default_start_pos_populates_correctly():
    board = Board()
    # Check pawns rows
    for j in range(8):
        p = board.board[1, j]
        assert p.get_name() == "pawn"
        assert p.team == "white"
        p = board.board[6, j]
        assert p.get_name() == "pawn"
        assert p.team == "black"
    # Check knights
    assert isinstance(board.board[0, 1], Knight)
    assert board.board[0, 1].team == "white"
    assert isinstance(board.board[0, 6], Knight)
    assert board.board[0, 6].team == "white"
    assert isinstance(board.board[7, 1], Knight)
    assert board.board[7, 1].team == "black"
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[7, 6].team == "black"
    # Check king and queen positions
    assert board.board[0, 4].get_name() == "king"
    assert board.board[0, 4].team == "white"
    assert board.board[0, 3].get_name() == "queen"
    assert board.board[0, 3].team == "white"
    assert board.board[7, 4].get_name() == "king"
    assert board.board[7, 4].team == "black"
    assert board.board[7, 3].get_name() == "queen"
    assert board.board[7, 3].team == "black"

def test_board_str_contains_rank_and_file_labels_and_pieces():
    board = Board()
    s = str(board)
    # Contains ranks (1-8) and files (A-H)
    for n in range(1, 9):
        assert str(n) in s
    for c in "ABCDEFGH":
        assert c in s
    # Contains some piece symbols like 'wk ' for white knight, 'bk ' for black knight
    assert "wk " in s or "bk " in s
    # Should be multiple lines
    assert s.count("\n") > 8

def test_board_update_board_basic_move_changes_positions_and_turn():
    board = Board()
    board.turn = "white"
    start = (1, 0)
    end = (2, 0)
    piece_before = board.board[start]
    assert piece_before.get_name() == "pawn"
    board.update_board((start, end, "base"), "white")
    # The piece moves
    piece_after = board.board[end]
    assert piece_after.get_name() == "pawn"
    assert piece_after.team == "white"
    # Start is now Dummy
    assert isinstance(board.board[start], Dummy)
    # Piece moved flag set
    assert piece_after.moved is True
    # turn flipped
    assert board.turn == "black"

def test_board_update_board_enpasse_removes_correct_pawn():
    board = Board()
    board.turn = "white"
    # Setup pawns in position for en passant
    board.board[4, 4] = board.board[1, 4]
    board.board[4, 4].set_position((4, 4))
    board.board[1, 4] = Dummy((1, 4))
    board.board[4, 4].moved = False

    board.board[4, 5] = board.board[6, 5]
    board.board[4, 5].set_position((4, 5))
    board.board[6, 5] = Dummy((6, 5))
    board.board[4, 5].moved = False
    # White pawn does en passant capture on black pawn
    start = (4, 4)
    end = (5, 5)
    board.update_board((start, end, "enpasse"), "white")
    # The capturing pawn has moved
    moved_pawn = board.board[end]
    assert moved_pawn.get_name() == "pawn"
    assert moved_pawn.team == "white"
    assert moved_pawn.moved is True
    # The original square is dummy
    assert isinstance(board.board[start], Dummy)
    # The captured pawn is removed (square (4,5))
    assert isinstance(board.board[(4, 5)], Dummy)
    # Turn flipped
    assert board.turn == "black"

def test_board_update_board_castling_moves_rook_and_king():
    board = Board()
    board.turn = "white"
    # Setup rooks and king in original pos for white
    # King from e1 (0,4) to g1 (0,6) for kingside castling
    start = (0, 4)
    end = (0, 6)
    board.update_board((start, end, "castling"), "white")
    # King moved
    king = board.board[end]
    assert king.get_name() == "king"
    assert king.team == "white"
    assert king.moved
    # Rook moved to f1 (0,5)
    rook = board.board[(0,5)]
    assert rook.get_name() == "rook"
    assert rook.team == "white"
    assert rook.moved
    # old king pos is dummy
    assert isinstance(board.board[start], Dummy)
    # old rook pos is dummy
    assert isinstance(board.board[(0,7)], Dummy)

def test_board_update_board_evolve_promotes_to_queen():
    board = Board()
    board.turn = "white"
    start = (1, 0)
    end = (7, 0)
    # put a white pawn at (1,0)
    pawn = board.board[1, 0]
    assert pawn.get_name() == "pawn"
    # move pawn to end with evolve
    board.update_board((start, end, "evolve"), "white")
    piece = board.board[end]
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # start square is dummy
    assert isinstance(board.board[start], Dummy)
    # turn changed to black
    assert board.turn == "black"

def test_board_copy_returns_deepcopy_and_is_equal():
    board1 = Board()
    board2 = board1.copy()
    assert board1.equal(board2)
    # Modify board2 and check not equal
    board2.board[1,0] = Dummy((1,0))
    assert not board1.equal(board2)

def test_board_equal_returns_true_for_identical_boards_and_false_for_different():
    board1 = Board()
    board2 = Board()
    assert board1.equal(board2)
    board2.board[0,0] = Dummy((0,0))
    assert board1.equal(board2) is False

def test_knight_get_moves_returns_correct_positions_and_type():
    board = Board()
    # Place white knight somewhere in middle to test moves
    knight = Knight((4, 4), "white")
    board.board[4, 4] = knight
    board_state = [board.copy()]
    moves = knight.get_moves(board_state)
    # Positions for knight in middle have up to 8 moves
    expected_positions = [
        (2, 3), (2, 5),
        (3, 2), (5, 2),
        (6, 3), (6, 5),
        (5, 6), (3, 6),
    ]
    result_positions = [move[1] for move in moves]
    for pos in expected_positions:
        # Because of pawns we might not get all moves, but no invalid positions
        assert pos in result_positions or board.board[pos].team == knight.team

def test_knight_get_symbol_and_name_and_value_for_teams():
    white_knight = Knight((0,0), "white")
    black_knight = Knight((0,0), "black")
    assert white_knight.get_symbol() == "wk "
    assert black_knight.get_symbol() == "bk "
    assert white_knight.get_name() == "knight"
    assert white_knight.get_value() == 3
    assert black_knight.get_name() == "knight"
    assert black_knight.get_value() == 3

def test_knight_copy_returns_new_instance_with_same_attributes():
    knight = Knight((3,3), "white")
    knight.moved = True
    knight_copy = knight.copy()
    assert isinstance(knight_copy, Knight)
    assert knight_copy.team == knight.team
    assert knight_copy.pos == knight.pos
    assert knight_copy.moved == knight.moved
    # They should not be the same object
    assert knight_copy is not knight

def test_gameengine_initializes_with_default_players_and_board_correctly():
    board = Board()
    engine = GameEngine(board)
    assert engine.Board is board
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.move_counter == 0
    assert isinstance(engine.game_states, list)
    assert isinstance(engine.game_states[0], Board)
    assert engine.black_check is False
    assert engine.white_check is False

def test_gameengine_get_turn_returns_correct_team_and_player():
    engine = GameEngine(Board())
    engine.move_counter = 0
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

def test_gameengine_num_alph_correctly_maps_moves():
    engine = GameEngine(Board())
    move_dict = {
        1: ((1, 0), (3, 1), "base"),
        2: ((6, 7), (5, 5), "base")
    }
    result = engine.num_alph(move_dict)
    assert result[1] == ["A2", "B4"]
    assert result[2] == ["H7", "F6"]

def test_gameengine_compare_states_returns_true_for_equal_boards():
    board1 = Board()
    board2 = board1.copy()
    engine = GameEngine(board1)
    assert engine.compare_states(board1, board2) is True
    board2.board[0,0] = Dummy((0,0))
    assert engine.compare_states(board1, board2) is False
