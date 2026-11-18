import pytest
import numpy as np
import random
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy

# Note: random is imported because we must seed it in tests that depend on any randomness in tested code.
# The code under test uses no visible explicit randomness in these parts, but requirement is to seed if used.
# Defensive seeding done anyway where relevant.


def test_clean_board_returns_8x8_array_of_Dummy():
    random.seed(0)
    board_array = clean_board()
    assert isinstance(board_array, np.ndarray)
    assert board_array.shape == (8, 8)
    # Check all positions are Dummy instances and with proper positions assigned
    for i in range(8):
        for j in range(8):
            piece = board_array[i, j]
            assert isinstance(piece, Dummy)
            assert piece.pos == (i, j)
            # Team should be None or Dummy's default team (None)
            assert getattr(piece, "team", None) is None


def test_board_default_starts_with_proper_pieces_and_turn():
    random.seed(0)
    board = Board()
    # The turn should start as white
    assert board.turn == "white"
    # Check specific squares for expected piece types by symbol
    # white pieces on row 0 and 1
    expected_white_pieces = {
        (0, 0): "wr ",  # rook
        (0, 1): "wk ",  # knight
        (0, 2): "wb ",  # bishop
        (0, 3): "wQ ",  # queen
        (0, 4): "wK ",  # king
        (0, 5): "wb ",  # bishop
        (0, 6): "wk ",  # knight
        (0, 7): "wr ",  # rook
    }
    for j in range(8):
        # Pawns at row 1 white
        assert board.board[1, j].get_symbol() == "wp "
    for pos, symbol in expected_white_pieces.items():
        assert board.board[pos].get_symbol() == symbol

    # black pieces on rows 6 and 7
    expected_black_pieces = {
        (7, 0): "br ",  # rook
        (7, 1): "bk ",  # knight
        (7, 2): "bb ",  # bishop
        (7, 3): "bQ ",  # queen
        (7, 4): "bK ",  # king
        (7, 5): "bb ",  # bishop
        (7, 6): "bk ",  # knight
        (7, 7): "br ",  # rook
    }
    for j in range(8):
        # Pawns at row 6 black
        assert board.board[6, j].get_symbol() == "bp "
    for pos, symbol in expected_black_pieces.items():
        assert board.board[pos].get_symbol() == symbol


def test_board_str_should_contain_all_positions_and_symbols():
    random.seed(0)
    board = Board()
    board_str = str(board)
    # The string should have row labels 1 to 8, and column labels A to H
    for i in range(1, 9):
        assert str(i) in board_str
    for col in "ABCDEFGH":
        assert col in board_str
    # Some known pieces symbols should be present (without color codes)
    for i in range(8):
        for j in range(8):
            piece = board.board[i, j]
            # piece.symbol should be present ignoring ANSI colors
            # We can't directly check colored output, but symbol should be in output string as a substring
            assert piece.symbol.strip() in board_str


def test_board_update_board_base_move_changes_positions_and_turn(monkeypatch):
    random.seed(0)
    board = Board()
    start_pos = (1, 0)  # white pawn initial position
    end_pos = (3, 0)    # two steps forward pawn move

    # We patch Dummy to count number of times it is called to check replacement in update_board
    dummy_calls = []

    original_dummy = type(board.board[0, 0])

    def dummy_side_effect(pos):
        dummy_calls.append(pos)
        return original_dummy(pos, None)

    # monkeypatch dummy for the purpose of update_board test on board.board assignment to Dummy
    # but since Dummy is constructed inside darkchess.src.board, we can't monkeypatch external Dummy directly safely here.
    # Instead, just test effects on board.

    move = (start_pos, end_pos, None)  # key_word None means 'base'
    current_turn = board.turn
    board.update_board(move, "white")

    # Check starting pos is Dummy now
    assert isinstance(board.board[start_pos], Dummy)
    # Check end position is Pawn piece of team white (moved)
    end_piece = board.board[end_pos]
    assert end_piece.team == "white"
    # updated position
    assert end_piece.pos == end_pos
    # moved flag set True
    assert getattr(end_piece, "moved", True) is True
    # turn should flip to black now
    assert board.turn != current_turn
    # now black's turn
    assert board.turn == "black"


def test_board_update_board_enpasse_removes_pawn_behind(monkeypatch):
    random.seed(0)
    board = Board()

    # Setup a minimal scenario to test en passant move:
    # White pawn moved from (4,4) to (5,5) capturing black pawn at (4,5) en passant
    # Place white pawn at (4, 4) and black pawn at (4, 5)
    from darkchess.src.pawn import Pawn

    board.board = clean_board()
    wp = Pawn((4, 4), "white")
    bp = Pawn((4, 5), "black")
    board.board[4, 4] = wp
    board.board[4, 5] = bp

    move = ((4, 4), (5, 5), "enpasse")
    board.turn = "white"
    board.update_board(move, "white")

    # White pawn should be at (5,5)
    piece_at_end = board.board[5, 5]
    assert piece_at_end is not None
    assert piece_at_end.team == "white"
    # The starting position should be Dummy
    assert isinstance(board.board[4, 4], Dummy)
    # The captured pawn at (4,5) should now be Dummy (because en passant captures the pawn horizontally adjacent)
    assert isinstance(board.board[(4, 5)], Dummy)
    # Turn toggled to black
    assert board.turn == "black"


def test_board_update_board_castling_king_side(monkeypatch):
    random.seed(0)
    board = Board()
    # Set up a basic board for castling right side for white
    board.board = clean_board()
    from darkchess.src.king import King
    from darkchess.src.rook import Rook

    # Position white king and rook at their initial positions for castling
    k_pos = (0, 4)
    r_pos = (0, 7)
    board.board[k_pos] = King(k_pos, "white")
    board.board[r_pos] = Rook(r_pos, "white")

    # Castling move king to (0,6)
    move = (k_pos, (0, 6), "castling")
    board.turn = "white"
    board.update_board(move, "white")

    # King should be at (0,6) now and moved True
    king_piece = board.board[(0, 6)]
    assert king_piece.get_name() == "king"
    assert king_piece.pos == (0, 6)
    assert king_piece.moved is True

    # Rook should be moved from (0,7) to (0,5) and moved True
    rook_piece = board.board[(0, 5)]
    assert rook_piece.get_name() == "rook"
    assert rook_piece.pos == (0, 5)
    assert rook_piece.moved is True

    # Positions (0,4) and (0,7) should be Dummy now
    assert isinstance(board.board[(0, 4)], Dummy)
    assert isinstance(board.board[(0, 7)], Dummy)

    # Turn toggled to black
    assert board.turn == "black"


def test_board_update_board_castling_queen_side(monkeypatch):
    random.seed(0)
    board = Board()
    # Set up a basic board for queen side castling for black
    board.board = clean_board()
    from darkchess.src.king import King
    from darkchess.src.rook import Rook

    k_pos = (7, 4)
    r_pos = (7, 0)
    board.board[k_pos] = King(k_pos, "black")
    board.board[r_pos] = Rook(r_pos, "black")

    move = (k_pos, (7, 2), "castling")
    board.turn = "black"
    board.update_board(move, "black")

    # King at (7, 2), moved True
    king_piece = board.board[(7, 2)]
    assert king_piece.get_name() == "king"
    assert king_piece.pos == (7, 2)
    assert king_piece.moved is True

    # Rook at (7, 3), moved True
    rook_piece = board.board[(7, 3)]
    assert rook_piece.get_name() == "rook"
    assert rook_piece.pos == (7, 3)
    assert rook_piece.moved is True

    # Old positions Dummy
    assert isinstance(board.board[(7, 4)], Dummy)
    assert isinstance(board.board[(7, 0)], Dummy)

    # Turn toggled to white
    assert board.turn == "white"


def test_board_update_board_evolve_promotes_pawn_to_queen(monkeypatch):
    random.seed(0)
    board = Board()
    # Place a white pawn ready to promote
    from darkchess.src.pawn import Pawn
    board.board = clean_board()
    pos_start = (6, 0)
    pos_end = (7, 0)
    board.board[pos_start] = Pawn(pos_start, "white")
    board.turn = "white"

    # Make evolution move
    move = (pos_start, pos_end, "evolve")
    board.update_board(move, "white")

    # The end pos should now have a Queen (symbol starting with "wQ ")
    piece = board.board[pos_end]
    # There is no Queen import for symbols here, so test symbol string startswith wQ
    piece_sym = piece.symbol
    assert piece_sym.startswith("wQ")
    # The start pos should be Dummy again
    assert isinstance(board.board[pos_start], Dummy)
    # Turn toggled
    assert board.turn == "black"


def test_board_copy_is_deep_copy_independent_of_original():
    random.seed(0)
    board = Board()
    board_copy = board.copy()

    # They should not be the same object
    assert board is not board_copy
    # Board contents different objects
    assert board.board is not board_copy.board
    # Pieces at each position differ objects with same position & symbol
    for i in range(8):
        for j in range(8):
            orig_piece = board.board[i, j]
            copied_piece = board_copy.board[i, j]
            assert orig_piece is not copied_piece
            # Position is reset correctly
            assert copied_piece.pos == (i, j)
            # Symbols equal
            assert orig_piece.symbol == copied_piece.symbol

    # Turn copied correctly
    assert board.turn == board_copy.turn

    # Modifying original piece moved attribute doesn't modify copy
    orig_pawn = board.board[1, 0]
    copy_pawn = board_copy.board[1, 0]
    orig_pawn.moved = True
    assert getattr(copy_pawn, "moved", False) is False


def test_board_equal_confirms_board_states_correctly():
    random.seed(0)
    board1 = Board()
    board2 = board1.copy()

    assert board1.equal(board2) is True

    # Modify one piece symbol in board2 by replacing piece at (1,1) with Dummy instead pawn, should differ
    board2.board[1, 1] = Dummy((1, 1))
    assert board1.equal(board2) is False

    # Modify piece's symbol in board2 at (1,0) to different symbol forcibly to break equality
    # Not typically possible, but simulate with patch
    class FakePiece:
        def __init__(self):
            self.symbol = "XX "

    board2.board[1, 0] = FakePiece()
    assert board1.equal(board2) is False


def test_knight_get_name_value_symbol_and_copy():
    random.seed(0)
    pos = (3, 3)
    kt_white = Knight(pos, "white")
    kt_black = Knight(pos, "black")

    assert kt_white.get_name() == "knight"
    assert kt_black.get_name() == "knight"

    assert kt_white.get_value() == 3
    assert kt_black.get_value() == 3

    assert kt_white.get_symbol() == "wk "
    assert kt_black.get_symbol() == "bk "

    # Test invalid team raises
    kt_weird = Knight(pos, "green")
    with pytest.raises(ValueError):
        kt_weird.get_symbol()

    # Test copy creates a separate but equal piece
    kt_white.moved = True
    copy_kt = kt_white.copy()
    assert copy_kt.get_name() == "knight"
    assert copy_kt.get_symbol() == "wk "
    assert copy_kt.pos == pos
    assert copy_kt.moved == kt_white.moved
    assert copy_kt is not kt_white


def test_knight_get_moves_correct_for_positions():
    random.seed(0)
    # Setup board with clean board + one knight at center for white and black and test moves
    base_board = Board()
    board_array = clean_board()
    knight_pos = (4, 4)

    # Place white knight in center
    white_knight = Knight(knight_pos, "white")
    board_array[white_knight.pos] = white_knight
    # Set some friendly pieces blocking knight for white and some enemy pieces
    from darkchess.src.pawn import Pawn

    # friendly pieces around knight to block some moves for white
    blocking_pos_white = [(2, 3), (3, 2)]
    for pos in blocking_pos_white:
        board_array[pos] = Pawn(pos, "white")

    # enemy pieces around knight for white (should be capturable)
    enemy_pos_white = [(2, 5), (3, 6)]
    for pos in enemy_pos_white:
        board_array[pos] = Pawn(pos, "black")

    # Prepare game_states list with one element (current board)
    game_states = [Board()]
    game_states[0].board = board_array

    moves = white_knight.get_moves(game_states)
    # Moves must not include positions occupied by friendly pieces
    for move in moves:
        _, end_pos, _ = move
        assert end_pos not in blocking_pos_white
    # moves must include enemy positions
    move_ends = [m[1] for m in moves]
    for pos in enemy_pos_white:
        assert pos in move_ends
    # Moves must be in legal knight move offsets
    knight_offsets = {(2, 1), (2, -1), (-2, 1), (-2, -1),
                      (1, 2), (1, -2), (-1, 2), (-1, -2)}
    for _, end, _ in moves:
        dx = end[0] - knight_pos[0]
        dy = end[1] - knight_pos[1]
        assert (dx, dy) in knight_offsets

    # Now test black knight similarly with white pieces in some blocking
    board_array = clean_board()
    black_knight_pos = (4, 4)
    black_knight = Knight(black_knight_pos, "black")
    board_array[black_knight_pos] = black_knight

    # friendly blocking black knight
    blocking_pos_black = [(6, 5), (5, 6)]
    # enemy positions for black
    enemy_pos_black = [(6, 3), (5, 2)]

    for pos in blocking_pos_black:
        board_array[pos] = Pawn(pos, "black")
    for pos in enemy_pos_black:
        board_array[pos] = Pawn(pos, "white")

    game_states = [Board()]
    game_states[0].board = board_array
    moves_black = black_knight.get_moves(game_states)

    # Moves must exclude blocking friendly, include enemies
    ends_black = [m[1] for m in moves_black]
    for pos in blocking_pos_black:
        assert pos not in ends_black
    for pos in enemy_pos_black:
        assert pos in ends_black
    # Moves offsets must be knight moves
    for move in moves_black:
        _, end, _ = move
        dx = end[0] - black_knight_pos[0]
        dy = end[1] - black_knight_pos[1]
        assert (dx, dy) in knight_offsets


def test_gameengine_get_turn_alternates_between_white_and_black():
    random.seed(0)
    board = Board()
    engine = GameEngine(board)
    # move_counter initial zero means white turn and player1 mode human
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"

    engine.move_counter = 1
    team, player = engine.get_turn()
    assert team == "black"
    assert player == "human"

    # With non-default players
    engine = GameEngine(board, player1="human", player2="computer")
    engine.move_counter = 0
    team, player = engine.get_turn()
    assert (team, player) == ("white", "human")
    engine.move_counter = 1
    team, player = engine.get_turn()
    assert (team, player) == ("black", "computer")


def test_gameengine_num_alph_converts_correctly():
    random.seed(0)
    # Positions using 0-based indexing converted to chess notation eg (row,col) to Letter and row+1
    move_dict = {
        1: ((1, 0), (2, 2), "base"),
        2: ((6, 7), (5, 5), "base"),
    }
    expected = {
        1: ["A2", "C3"],  # row=1 col=0 is A2, row=2 col=2 is C3
        2: ["H7", "F6"],  # row=6 col=7 is H7, row=5 col=5 is F6
    }
    result = GameEngine.num_alph(move_dict)
    assert result == expected


def test_gameengine_undo_move_restores_previous_board_state():
    random.seed(0)
    board = Board()
    engine = GameEngine(board)
    initial_state = engine.Board.copy()
    # Make a valid move
    move = ((1, 0), (2, 0), "base")
    engine.Board.update_board(move, "white")
    engine.game_states.append(engine.Board.copy())
    engine.move_counter = 1

    # Undo move should revert to initial state
    engine.undo_move()
    assert engine.Board.equal(initial_state)
    assert engine.move_counter == 1  # move_counter does not change here


def test_gameengine_compare_states_true_for_equal_boards():
    random.seed(0)
    b1 = Board()
    b2 = b1.copy()
    engine = GameEngine(b1)
    assert engine.compare_states(b1, b2) is True

    # Modify b2 board by changing a piece symbol forcibly
    b2.board[0, 0] = Dummy((0, 0))
    assert engine.compare_states(b1, b2) is False
