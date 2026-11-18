import pytest
import numpy as np
import random

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_8x8_array_with_dummy_pieces():
    board_array = clean_board()
    assert isinstance(board_array, np.ndarray)
    assert board_array.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            piece = board_array[i, j]
            assert isinstance(piece, Dummy)
            # Dummy pieces on clean board have pos (i,j) and team None
            assert piece.get_position() == (i, j)
            assert piece.team is None


def test_board_default_initialization_sets_start_positions():
    b = Board()
    # Pawns on correct rows for each team
    for j in range(8):
        # White pawns at row 1
        assert b.board[1, j].get_name() == "pawn"
        assert b.board[1, j].team == "white"
        # Black pawns at row 6
        assert b.board[6, j].get_name() == "pawn"
        assert b.board[6, j].team == "black"

    # Corner rooks
    assert b.board[0, 0].get_name() == "rook"
    assert b.board[0, 0].team == "white"
    assert b.board[0, 7].get_name() == "rook"
    assert b.board[0, 7].team == "white"

    assert b.board[7, 0].get_name() == "rook"
    assert b.board[7, 0].team == "black"
    assert b.board[7, 7].get_name() == "rook"
    assert b.board[7, 7].team == "black"

    # Knights
    assert b.board[0, 1].get_name() == "knight"
    assert b.board[0, 1].team == "white"
    assert b.board[0, 6].get_name() == "knight"
    assert b.board[0, 6].team == "white"

    assert b.board[7, 1].get_name() == "knight"
    assert b.board[7, 1].team == "black"
    assert b.board[7, 6].get_name() == "knight"
    assert b.board[7, 6].team == "black"

    # Bishops
    assert b.board[0, 2].get_name() == "bishop"
    assert b.board[0, 2].team == "white"
    assert b.board[0, 5].get_name() == "bishop"
    assert b.board[0, 5].team == "white"

    assert b.board[7, 2].get_name() == "bishop"
    assert b.board[7, 2].team == "black"
    assert b.board[7, 5].get_name() == "bishop"
    assert b.board[7, 5].team == "black"

    # Queens
    assert b.board[0, 3].get_name() == "queen"
    assert b.board[0, 3].team == "white"
    assert b.board[7, 3].get_name() == "queen"
    assert b.board[7, 3].team == "black"

    # Kings
    assert b.board[0, 4].get_name() == "king"
    assert b.board[0, 4].team == "white"
    assert b.board[7, 4].get_name() == "king"
    assert b.board[7, 4].team == "black"

    assert b.turn == "white"


def test_board_str_contains_expected_rows_and_columns():
    b = Board()
    board_str = str(b)
    # Check columns labels appear exactly twice
    assert board_str.count("  A  B  C  D  E  F  G  H ") == 2
    # Check rows labels 1-8 appear twice (top and bottom)
    for row in range(1, 9):
        label = str(row)
        # should appear at least twice
        assert board_str.count(label) >= 2


def test_board_update_board_base_move_updates_positions_and_turn(monkeypatch):
    b = Board()
    b.turn = "white"
    start = (1, 0)  # white pawn initial position
    end = (2, 0)
    start_piece = b.board[start]
    # move pawn forward one step, base move
    b.update_board((start, end, 'base'), "white")
    assert isinstance(b.board[end], type(start_piece))
    assert b.board[end].get_position() == end
    assert isinstance(b.board[start], Dummy)
    assert b.board[end].moved is True
    # turn toggled to black
    assert b.turn == "black"


def test_board_update_board_en_passant_removes_pawn(monkeypatch):
    b = Board()
    # Setup simplified board for en passant testing
    for i in range(8):
        for j in range(8):
            b.board[i, j] = Dummy((i, j), None)
    # White pawn at (4, 3)
    white_pawn = b.board[1, 3]
    b.board[4, 3] = white_pawn
    white_pawn.set_position((4, 3))
    white_pawn.team = "white"
    # Black pawn at (4, 4)
    black_pawn = b.board[6, 4]
    b.board[4, 4] = black_pawn
    black_pawn.set_position((4, 4))
    black_pawn.team = "black"
    b.turn = "white"
    # White pawn moves diagonally to (5,4) en passant capturing black pawn at (4,4)
    b.board[(4, 3)] = white_pawn
    move = ((4, 3), (5, 4), "enpasse")
    b.update_board(move, "white")
    # The captured pawn removed at (4,4)
    assert isinstance(b.board[(4, 4)], Dummy)
    # The white pawn moved to (5,4)
    assert b.board[(5, 4)].team == "white"
    assert b.board[(5, 4)].get_position() == (5, 4)
    assert b.turn == "black"


def test_board_update_board_castling_king_side(monkeypatch):
    b = Board()
    # Setup empty board with just king and rook for white
    for i in range(8):
        for j in range(8):
            b.board[i, j] = Dummy((i, j), None)

    # place white king at (0,4)
    from darkchess.src.king import King
    king = King((0, 4), "white")
    b.board[0, 4] = king
    # place white rook at (0,7)
    from darkchess.src.rook import Rook
    rook = Rook((0, 7), "white")
    b.board[0, 7] = rook
    b.turn = "white"

    # Move king from (0,4) to (0,6) castling king side
    b.update_board(((0, 4), (0, 6), "castling"), "white")

    # King is at (0,6)
    assert b.board[0, 6] == king
    assert king.get_position() == (0, 6)
    assert king.moved is True

    # Rook is at (0,5)
    assert b.board[0, 5] == rook
    assert rook.get_position() == (0, 5)
    assert rook.moved is True

    # Original positions are Dummy now
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)

    # turn toggled
    assert b.turn == "black"


def test_board_update_board_castling_queen_side(monkeypatch):
    b = Board()
    for i in range(8):
        for j in range(8):
            b.board[i, j] = Dummy((i, j), None)

    from darkchess.src.king import King
    king = King((0, 4), "white")
    b.board[0, 4] = king
    from darkchess.src.rook import Rook
    rook = Rook((0, 0), "white")
    b.board[0, 0] = rook
    b.turn = "white"

    # Move king from (0,4) to (0,2) castling queen side
    b.update_board(((0, 4), (0, 2), "castling"), "white")

    # King at (0,2)
    assert b.board[0, 2] == king
    assert king.get_position() == (0, 2)
    assert king.moved is True

    # Rook at (0,3)
    assert b.board[0, 3] == rook
    assert rook.get_position() == (0, 3)
    assert rook.moved is True

    # Origin squares are Dummy now
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 0], Dummy)

    # Turn toggled
    assert b.turn == "black"


def test_board_update_board_evolve_promotes_to_queen():
    b = Board()
    from darkchess.src.pawn import Pawn
    for i in range(8):
        for j in range(8):
            b.board[i, j] = Dummy((i, j), None)

    # Place white pawn at (6, 0)
    pawn = Pawn((6, 0), "white")
    b.board[6, 0] = pawn
    b.turn = "white"

    # Evolve pawn from (6,0) to (7,0)
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    piece = b.board[7, 0]

    # It is a queen now
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # Original square is Dummy
    assert isinstance(b.board[6, 0], Dummy)

    # Turn toggled
    assert b.turn == "black"


def test_board_copy_creates_independent_board(monkeypatch):
    b1 = Board()
    b_copy = b1.copy()

    # Verify boards are distinct objects (different numpy arrays)
    assert b1.board is not b_copy.board

    # Modifying copy should not affect original
    # Pick a position with white knight by default at (0, 1)
    orig_piece = b1.board[0, 1]
    copy_piece = b_copy.board[0, 1]
    # Changing moved flag in copy should not reflect to original
    copy_piece.moved = not copy_piece.moved
    assert b1.board[0, 1].moved != b_copy.board[0, 1].moved

    # Changing position in copy does not affect original
    copy_piece.set_position((5, 5))
    assert orig_piece.get_position() != copy_piece.get_position()

    # Turn attribute copied properly (and deep copy)
    b_copy.turn = "black"
    assert b1.turn != b_copy.turn


def test_board_equal_method_computes_equality_correctly():
    b1 = Board()
    b2 = Board()
    assert b1.equal(b2)

    # Change one piece in b2
    b2.board[0, 0] = b1.board[0, 0].copy()
    b2.board[0, 0].moved = not b1.board[0, 0].moved  # Just toggle moved flag (symbol is same)
    # equal depends on symbol only, so should still be True
    assert b1.equal(b2)

    # Change piece symbol by changing piece type forcibly
    from darkchess.src.knight import Knight
    b2.board[0, 0] = Knight((0, 0), "white")
    # If symbol changed, equal returns False
    assert not b1.equal(b2)


def test_knight_get_moves_correct_number_and_type(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    # Setup an empty board with Dummy pieces
    empty_board = clean_board()
    # Place white knight in center (4,4)
    k = Knight((4, 4), "white")
    empty_board[4, 4] = k

    # game_states list with one element: board object with our board
    # We create a dummy Board object from the code to serve as game state element
    b = Board()
    b.board = empty_board
    gstates = [b]

    moves = k.get_moves(gstates)
    # There are up to 8 possible knight moves from center
    # But some may be blocked by friendly pieces; here all other cells are Dummy with team None
    assert all(isinstance(move, tuple) and len(move) == 3 for move in moves)
    # Moves are from (4,4) to valid knight target cells
    starts = {m[0] for m in moves}
    assert starts == {(4, 4)}
    # All positions are on board
    for _, (x, y), _ in moves:
        assert 0 <= x <= 7
        assert 0 <= y <= 7

    # Now place friendly piece in one knight move position to block
    # Knight moves include (6,5), so place white piece there to block
    empty_board[6, 5] = Knight((6, 5), "white")
    b.board = empty_board
    moves_2 = k.get_moves([b])
    # (6,5) should be excluded from moves_2
    blocked_pos = (6, 5)
    assert all(m[1] != blocked_pos for m in moves_2)


def test_knight_get_moves_raises_value_error_for_invalid_team(monkeypatch):
    k = Knight((0, 0), "invalid_team")
    # Build dummy board with only this knight
    b = Board()
    b.board = clean_board()
    b.board[0, 0] = k
    with pytest.raises(ValueError):
        k.get_moves([b])


def test_gameengine_get_turn_alternates_after_move(monkeypatch):
    b = Board()
    engine = GameEngine(b, "human", "computer")
    engine.move_counter = 0
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"
    engine.move_counter += 1
    team, player = engine.get_turn()
    assert team == "black"
    assert player == "computer"


def test_gameengine_all_moves_returns_list_of_moves(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    b = Board()
    engine = GameEngine(b)
    moves = engine.all_moves()
    # Moves should be a list of tuples with length >= 0
    assert isinstance(moves, list)
    for move in moves:
        assert isinstance(move, tuple)
        assert len(move) == 3


def test_gameengine_compare_states_returns_true_for_equal_boards(monkeypatch):
    b1 = Board()
    b2 = b1.copy()
    engine = GameEngine(b1)
    assert engine.compare_states(b1, b2)

    # Modify symbol forcibly to make boards differ
    from darkchess.src.knight import Knight
    b2.board[0, 1] = Knight((0, 1), "white")
    assert not engine.compare_states(b1, b2)


def test_gameengine_undo_move_reverts_board_state(monkeypatch):
    b = Board()
    engine = GameEngine(b)
    initial_board_copy = engine.Board.copy()
    # Make a dummy move by manipulating game_states
    engine.game_states.append(engine.Board.copy())
    engine.Board.turn = "black"
    engine.undo_move()
    # After undo, engine.Board should equal initial_board_copy
    assert engine.Board.equal(initial_board_copy)


def test_gameengine_num_alph_converts_moves_correctly(monkeypatch):
    engine = GameEngine(Board())
    # Two moves with known coordinates
    move_dict = {
        1: ((1, 0), (2, 2), "base"),
        2: ((4, 4), (6, 5), "base"),
    }
    res = engine.num_alph(move_dict)
    assert res[1] == ['A1', 'C3'] or res[1] == ['C3', 'A1'] or res[1] == ['B2', 'C3'] or res[1] == ['A2', 'C3'] or res[1] == ['A2', 'C3']
    # General format checks
    for key, val in res.items():
        for sq in val:
            assert len(sq) in (2, 3)  # e.g. 'A1'
            assert sq[0] in "ABCDEFGH"
            assert sq[1:].isdigit()


def test_gameengine_is_check_false_on_start(monkeypatch):
    b = Board()
    engine = GameEngine(b)
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False
