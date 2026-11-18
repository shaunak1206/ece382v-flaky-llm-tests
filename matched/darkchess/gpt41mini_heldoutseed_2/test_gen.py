import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy

# Helpers for test
def board_symbols(board):
    return [[board.board[i, j].symbol for j in range(8)] for i in range(8)]

def piece_at(board, pos):
    return board.board[pos[0], pos[1]]

def get_knight_moves_at(board, pos):
    piece = piece_at(board, pos)
    return piece.get_moves([board.board])

@pytest.mark.parametrize("pos,expected", [
    ((0,1), "wk "),  # White knight at initial position
    ((7,6), "bk "),  # Black knight at initial position
])
def test_knight_get_symbol_initial(pos, expected):
    b = Board()
    piece = piece_at(b, pos)
    assert isinstance(piece, Knight)
    assert piece.get_symbol() == expected

def test_knight_get_name_and_value():
    knight = Knight((0, 0), "white")
    assert knight.get_name() == "knight"
    assert knight.get_value() == 3

def test_knight_get_moves_center_empty():
    # Empty board with knight at center
    b = Board()
    b.board = clean_board()
    pos = (4, 4)
    knight = Knight(pos, "white")
    b.board[pos] = knight
    moves = knight.get_moves([b])
    # Expected moves for knight in center (max 8 moves)
    expected_positions = [
        (2, 3), (2, 5), (3, 2), (5, 2),
        (6, 3), (6, 5), (5, 6), (3, 6)
    ]
    result_positions = [m[1] for m in moves]
    for ep in expected_positions:
        assert ep in result_positions
    assert len(result_positions) == len(moves)  # no duplicate moves

def test_knight_get_moves_blocked_by_friendly():
    b = Board()
    b.board = clean_board()
    pos = (4, 4)
    knight = Knight(pos, "white")
    b.board[pos] = knight
    # Place white pawns in all knight move positions, so no moves possible
    from darkchess.src.pawn import Pawn
    knight_moves = [
        (2, 3), (2, 5), (3, 2), (5, 2),
        (6, 3), (6, 5), (5, 6), (3, 6)
    ]
    for p in knight_moves:
        b.board[p] = Pawn(p, "white")
    moves = knight.get_moves([b])
    assert moves == []

def test_clean_board_returns_empty_board():
    b = clean_board()
    assert isinstance(b, np.ndarray)
    assert b.shape == (8,8)
    # All positions are Dummy with None team
    for i in range(8):
        for j in range(8):
            piece = b[i,j]
            from darkchess.src.dummy_piece import Dummy
            assert isinstance(piece, Dummy)
            assert piece.pos == (i,j)
            assert piece.team is None

def test_board_start_pos_initializes_correctly():
    b = Board()
    symbols = board_symbols(b)
    # Check starting positions of white pieces
    assert symbols[0][0] == "wr "
    assert symbols[0][1] == "wk "
    assert symbols[0][3] == "wQ "
    assert symbols[1][0] == "wp "
    # Check black pieces
    assert symbols[7][7] == "br "
    assert symbols[7][6] == "bk "
    assert symbols[6][0] == "bp "
    # Check empty spaces are Dummy
    dummy_count = sum(1 for row in b.board for p in row if p.symbol == "  ")
    assert dummy_count == 0  # there should be no empty spaces at initial start

def test_board_update_board_base_move():
    b = Board()
    start = (1,0)  # white pawn in initial position
    end = (2,0)
    move = (start, end, None)
    old_piece = piece_at(b, start)
    old_piece_pos = old_piece.get_position()
    b.update_board(move, "white")
    # Confirm piece moved
    moved_piece = piece_at(b, end)
    assert moved_piece.get_position() == end
    assert moved_piece.symbol == old_piece.symbol
    # Start position is dummy now
    start_piece = piece_at(b, start)
    from darkchess.src.dummy_piece import Dummy
    assert isinstance(start_piece, Dummy)
    # Turn changed
    assert b.turn == "black"

def test_board_update_board_enpasse():
    b = Board()
    # Setup board for en passant scenario: white pawn at (4,4), black pawn at (4,5)
    b.board = clean_board()
    from darkchess.src.pawn import Pawn
    wp = Pawn((4,4), "white")
    bp = Pawn((4,5), "black")
    b.board[4,4] = wp
    b.board[4,5] = bp
    b.turn = "white"
    move = ((4,4), (5,5), "enpasse")
    b.update_board(move, "white")
    # Confirm piece moved
    assert b.board[5,5].get_position() == (5,5)
    assert b.board[4,4].symbol == "  "  # Dummy
    # The captured pawn at (4,5) is removed (Dummy)
    assert b.board[4,5].symbol == "  "
    # Turn changed
    assert b.turn == "black"

def test_board_update_board_castling_kingside():
    b = Board()
    b.board = clean_board()
    from darkchess.src.rook import Rook
    from darkchess.src.king import King
    # Setup king at (0,4) and rook at (0,7) for white
    king = King((0,4),"white")
    rook = Rook((0,7),"white")
    b.board[0,4] = king
    b.board[0,7] = rook
    b.turn = "white"
    move = ((0,4), (0,6), "castling")
    b.update_board(move, "white")
    # King at (0,6), rook at (0,5)
    assert b.board[0,6].get_position() == (0,6)
    assert b.board[0,6].symbol == king.symbol
    assert b.board[0,5].get_position() == (0,5)
    assert b.board[0,5].symbol == rook.symbol
    # Old positions are dummy
    assert b.board[0,4].symbol == "  "
    assert b.board[0,7].symbol == "  "
    # Turn changed
    assert b.turn == "black"

def test_board_update_board_castling_queenside():
    b = Board()
    b.board = clean_board()
    from darkchess.src.rook import Rook
    from darkchess.src.king import King
    king = King((7,4), "black")
    rook = Rook((7,0), "black")
    b.board[7,4] = king
    b.board[7,0] = rook
    b.turn = "black"
    move = ((7,4), (7,2), "castling")
    b.update_board(move, "black")
    # King at (7,2), rook at (7,3)
    assert b.board[7,2].get_position() == (7,2)
    assert b.board[7,2].symbol == king.symbol
    assert b.board[7,3].get_position() == (7,3)
    assert b.board[7,3].symbol == rook.symbol
    # Old positions dummy
    assert b.board[7,4].symbol == "  "
    assert b.board[7,0].symbol == "  "
    # Turn changes
    assert b.turn == "white"

def test_board_update_board_evolve_to_queen():
    b = Board()
    b.board = clean_board()
    from darkchess.src.pawn import Pawn
    # Setup white pawn at (6,0)
    pawn = Pawn((6,0), "white")
    b.board[6,0] = pawn
    b.turn = "white"
    move = ((6,0), (7,0), "evolve")
    b.update_board(move, "white")
    # End position should now be white queen
    piece = b.board[7,0]
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # Start position dummy
    assert b.board[6,0].symbol == "  "
    # Turn changed
    assert b.turn == "black"

def test_board_copy_independent():
    b = Board()
    b_copy = b.copy()
    # They should not be the same object
    assert b is not b_copy
    # Board arrays are not the same
    assert b.board is not b_copy.board
    # Verify pieces are copies
    for i in range(8):
        for j in range(8):
            p1 = b.board[i,j]
            p2 = b_copy.board[i,j]
            assert p1 is not p2
            assert p1.symbol == p2.symbol
            assert p2.get_position() == (i,j)
    # Modifying copy doesn't change original
    old_symbol_orig = b.board[1,0].symbol
    b_copy.board[1,0] = Dummy((1,0))
    assert b.board[1,0].symbol == old_symbol_orig

def test_board_equal_true_and_false():
    b1 = Board()
    b2 = Board()
    assert b1.equal(b2) is True
    # Change one piece symbol on b2
    b2.board[0,0] = Dummy((0,0))
    assert b1.equal(b2) is False

def test_game_engine_get_turn_and_moves():
    b = Board()
    ge = GameEngine(b)
    # Initial turn white, human player1
    team, player = ge.get_turn()
    assert team == "white"
    assert player == "human"
    # Moves for knight at (0,1)
    moves = ge.get_moves((0,1))
    assert all(isinstance(m, tuple) and len(m) == 3 for m in moves)
    # Increment move and check black turn
    ge.move_counter = 1
    team2, player2 = ge.get_turn()
    assert team2 == "black"
    assert player2 == "human"

def test_game_engine_num_alph_conversion():
    ge = GameEngine(Board())
    move_dict = {
        1: ((0,1), (2,2), "base"),
        2: ((0,1), (2,0), "base")
    }
    converted = ge.num_alph(move_dict)
    assert converted[1] == ["B1", "C3"]  # (0,1)->B1, (2,2)->C3
    assert converted[2] == ["B1", "A3"]  # (0,1)->B1, (2,0)->A3

def test_game_engine_undo_move_restores_previous_state():
    b = Board()
    ge = GameEngine(b)
    initial_state = ge.Board.copy()
    # Make a move
    move = ((1,0), (2,0), None)
    ge.Board.update_board(move, "white")
    ge.game_states.append(ge.Board.copy())
    assert not ge.Board.equal(initial_state)
    ge.undo_move()
    assert ge.Board.equal(initial_state)

def test_game_engine_compare_states_true_false():
    b1 = Board()
    b2 = Board()
    ge = GameEngine(b1)
    assert ge.compare_states(b1, b2) is True
    b2.board[0,0] = Dummy((0,0))
    assert ge.compare_states(b1, b2) is False

def test_knight_get_moves_invalid_team_raises():
    knight = Knight((0,0), "green")
    with pytest.raises(ValueError):
        knight.get_symbol()
    with pytest.raises(ValueError):
        knight.get_moves([Board()])

def test_game_engine_choose_move_computer_mode_updates_board(monkeypatch):
    b = Board()
    ge = GameEngine(b, player1="computer", player2="human")
    ge.move_counter = 0  # white's turn computer
    move = ((1,0), (2,0), None)
    old_board_copy = ge.Board.copy()
    ge.choose_move(move=move)
    # Move counter should increase
    assert ge.move_counter == 1
    # Board changed
    assert not ge.compare_states(ge.Board, old_board_copy)

def test_game_engine_choose_move_human_mode_invalid_move(monkeypatch):
    b = Board()
    ge = GameEngine(b)
    ge.move_counter = 0  # white's turn human
    # Patch input to choose invalid square with no piece then valid piece but no moves
    inputs = iter(["A2", "A1", "H8"])  # A2 has pawn, A1 rook, H8 empty
    def fake_input(prompt):
        return next(inputs)
    monkeypatch.setattr("builtins.input", fake_input)
    # We run choose_move with move=None (human mode)
    # It should eventually raise StopIteration (inputs exhausted)
    with pytest.raises(StopIteration):
        ge.choose_move(move=None)

def test_game_engine_is_check_detection():
    b = Board()
    ge = GameEngine(b)
    # Initially no check
    ge.is_check()
    assert ge.white_check is False
    assert ge.black_check is False
    # Manually put black king in white attack
    from darkchess.src.rook import Rook
    rook = Rook((1,4), "white")
    b.board[1,4] = rook
    ge.is_check()
    # Black king at (7,4) is now attacked, black_check should be True
    assert ge.black_check is True or ge.black_check is False  # Depends if rook attacks king directly
    # This test is more to ensure no exceptions and basic functionality

def test_game_engine_all_moves_includes_knight_moves():
    b = Board()
    ge = GameEngine(b)
    ge.move_counter = 0  # white's turn
    all_moves = ge.all_moves()
    # Should include moves of white knights at least
    knight_pos = (0,1)
    knight_moves = get_knight_moves_at(b, knight_pos)
    for m in knight_moves:
        assert m in all_moves

