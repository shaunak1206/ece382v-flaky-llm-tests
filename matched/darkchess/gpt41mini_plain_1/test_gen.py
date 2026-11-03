import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn


# --- Tests for Board ---

def test_board_initialization_default():
    board = Board()
    # Check board is 8x8 numpy array
    assert isinstance(board.board, np.ndarray)
    assert board.board.shape == (8, 8)
    # Check pieces in expected positions (example: white pawns at row 1)
    for j in range(8):
        assert board.board[1, j].get_name() == "pawn"
        assert board.board[1, j].team == "white"
    # Black pawns
    for j in range(8):
        assert board.board[6, j].get_name() == "pawn"
        assert board.board[6, j].team == "black"
    # Knights positions white
    assert isinstance(board.board[0,1], Knight)
    assert board.board[0,1].team == "white"
    assert isinstance(board.board[0,6], Knight)
    assert board.board[0,6].team == "white"
    # Knights positions black
    assert isinstance(board.board[7,1], Knight)
    assert board.board[7,1].team == "black"
    assert isinstance(board.board[7,6], Knight)
    assert board.board[7,6].team == "black"
    # Turn starts with white
    assert board.turn == "white"


def test_board_initialization_with_custom_board():
    custom_board = clean_board()
    custom_board[0,0] = Knight((0,0), "white")
    board = Board(starting_board=custom_board)
    # Should use custom_board as board reference
    assert isinstance(board.board[0,0], Knight)
    assert board.board[0,0].team == "white"
    # Turn should be white by default
    assert board.turn == "white"


def test_board_update_board_base_move():
    board = Board()
    start = (1,0)  # white pawn initial pos
    end = (2,0)
    original_piece = board.board[start]
    move = (start, end, "base")
    board.update_board(move, "white")
    # The piece should move
    assert board.board[end].get_name() == "pawn"
    assert board.board[end].team == "white"
    # The original position is now Dummy
    assert isinstance(board.board[start], Dummy)
    # The moved flag should be True
    assert board.board[end].moved is True
    # Turn switched to black
    assert board.turn == "black"


def test_board_update_board_enpasse():
    board = Board()
    # Setup enpasse scenario in simplified way:
    # White pawn at (4,1), black pawn at (6,2) moves two steps to (4,2)
    board.board = clean_board()
    white_pawn_pos = (4,1)
    black_pawn_start = (6,2)
    black_pawn_end = (4,2)

    from darkchess.src.pawn import Pawn
    board.board[4,1] = Pawn(white_pawn_pos, "white")
    board.board[6,2] = Pawn(black_pawn_start, "black")

    # Simulate black pawn moves two steps
    move = (black_pawn_start, black_pawn_end, "base")
    board.update_board(move, "black")

    # Now white pawn performs enpasse capture
    move_enpasse = (white_pawn_pos, black_pawn_end, "enpasse")
    board.update_board(move_enpasse, "white")

    # Black pawn should be captured (at (4,2) replaced with white pawn)
    p = board.board[black_pawn_end]
    assert p.team == "white"
    assert p.get_name() == "pawn"
    # The square of black pawn just captured should be Dummy
    assert isinstance(board.board[(black_pawn_end[0]+1, black_pawn_end[1])], Dummy)


def test_board_update_board_castling_kingside():
    board = Board()
    # Setup simplified castling scenario (white)
    # King is at (0,4), rook at (0,7)
    king_pos = (0,4)
    rook_pos = (0,7)
    king_castle_end = (0,6)

    # Ensure king and rook are at start position and moved==False initially
    king = board.board[king_pos]
    rook = board.board[rook_pos]
    king.moved = False
    rook.moved = False

    move = (king_pos, king_castle_end, "castling")
    board.update_board(move, "white")

    # King moved to (0,6)
    assert board.board[king_castle_end].get_name() == "king"
    assert board.board[king_castle_end].team == "white"
    assert board.board[king_castle_end].moved is True
    # Rook should be at (0,5)
    rook_after = board.board[(0,5)]
    assert rook_after.get_name() == "rook"
    assert rook_after.team == "white"
    assert rook_after.moved is True
    # Start and old rook positions should be Dummy
    assert isinstance(board.board[king_pos], Dummy)
    assert isinstance(board.board[rook_pos], Dummy)


def test_board_update_board_castling_queenside():
    board = Board()
    # Setup simplified castling (white queenside)
    king_pos = (0,4)
    rook_pos = (0,0)
    king_castle_end = (0,2)

    king = board.board[king_pos]
    rook = board.board[rook_pos]
    king.moved = False
    rook.moved = False

    move = (king_pos, king_castle_end, "castling")
    board.update_board(move, "white")

    # King moved to (0,2)
    assert board.board[king_castle_end].get_name() == "king"
    assert board.board[king_castle_end].team == "white"
    assert board.board[king_castle_end].moved is True

    # Rook should be at (0,3)
    rook_after = board.board[(0,3)]
    assert rook_after.get_name() == "rook"
    assert rook_after.team == "white"
    assert rook_after.moved is True

    # Start and old rook positions Dummy
    assert isinstance(board.board[king_pos], Dummy)
    assert isinstance(board.board[rook_pos], Dummy)


def test_board_update_board_evolve():
    board = Board()
    # White pawn reaches last row to evolve
    start = (6, 0)
    end = (7, 0)
    from darkchess.src.pawn import Pawn
    board.board[start] = Pawn(start, "white")
    move = (start, end, "evolve")
    board.update_board(move, "white")

    # New piece at end should be Queen (according to update_board logic)
    piece = board.board[end]
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # Start is now Dummy
    assert isinstance(board.board[start], Dummy)


def test_board_copy_and_equal_method():
    board = Board()
    board_copy = board.copy()

    # The boards should be equal initially
    assert board.equal(board_copy)

    # Modify copy and check inequality
    board_copy.board[1,1] = Dummy((1,1))
    assert not board.equal(board_copy)


def test_board_equal_with_different_symbols():
    board1 = Board()
    board2 = Board()
    # Change one piece in board2
    board2.board[0,0] = Dummy((0,0))
    assert not board1.equal(board2)


def test_board_str_and_view_methods(capsys):
    board = Board()
    s = str(board)
    assert isinstance(s, str)
    board.view_board()
    captured = capsys.readouterr()
    assert s in captured.out
    board.view_board_debug()
    captured_debug = capsys.readouterr()
    assert len(captured_debug.out) > 0


# --- Tests for Knight ---

def test_knight_get_name_and_value_and_symbol():
    knight_white = Knight((0,0), "white")
    knight_black = Knight((0,0), "black")

    assert knight_white.get_name() == "knight"
    assert knight_black.get_name() == "knight"

    assert knight_white.get_value() == 3
    assert knight_black.get_value() == 3

    assert knight_white.get_symbol() == "wk "
    assert knight_black.get_symbol() == "bk "

    with pytest.raises(ValueError):
        Knight((0,0), "green").get_symbol()


def make_knight_moves_board(knight_pos, knight_team):
    from darkchess.src.piece import Piece
    from darkchess.src.dummy_piece import Dummy

    # Create empty board with Dummies
    board = np.empty((8,8), dtype=Piece)
    for i in range(8):
        for j in range(8):
            board[i,j] = Dummy((i,j), None)

    # Place knight
    knight = Knight(knight_pos, knight_team)
    board[knight_pos] = knight

    # Place some enemy pieces in knight's move range
    if knight_team == "white":
        enemy_team = "black"
    else:
        enemy_team = "white"

    # Place a few enemy pawns at some knight moves
    possible_enemy_positions = [
        (knight_pos[0]-2, knight_pos[1]-1),
        (knight_pos[0]-1, knight_pos[1]-2),
        (knight_pos[0]+1, knight_pos[1]-2),
        (knight_pos[0]+2, knight_pos[1]-1)
    ]
    from darkchess.src.pawn import Pawn
    for pos in possible_enemy_positions:
        if 0 <= pos[0] <= 7 and 0 <= pos[1] <=7:
            board[pos] = Pawn(pos, enemy_team)

    return board, knight


def test_knight_get_moves_returns_valid_moves():
    board, knight = make_knight_moves_board((4,4), "white")
    game_states = [Board().copy()]
    # Overwrite latest state to contain our test board
    game_states[-1].board = board

    moves = knight.get_moves(game_states)
    # Each move should be from knight pos start
    for move in moves:
        start, end, move_type = move
        assert start == (4,4)
        assert move_type == "base"
        # End position should be in board limits
        assert 0 <= end[0] <= 7
        assert 0 <= end[1] <= 7


def test_knight_get_moves_blocked_by_friendly():
    board, knight = make_knight_moves_board((4,4), "white")
    # Place friendly piece where knight could move
    board[6,5] = Pawn((6,5), "white")
    game_states = [Board().copy()]
    game_states[-1].board = board

    moves = knight.get_moves(game_states)
    # That position (6,5) should not appear in any move ends
    for m in moves:
        assert m[1] != (6,5)


def test_knight_copy_creates_independent_piece():
    knight1 = Knight((2,3), "black")
    knight1.moved = True
    knight2 = knight1.copy()

    assert knight2.get_name() == "knight"
    assert knight2.team == "black"
    assert knight2.get_position() == (2,3)
    assert knight2.moved == True

    knight2.moved = False
    assert knight1.moved is True  # original not changed


# --- Tests for clean_board (utils) ---

def test_clean_board_creates_dummy_board():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8,8)
    from darkchess.src.dummy_piece import Dummy
    for i in range(8):
        for j in range(8):
            piece = board[i,j]
            assert isinstance(piece, Dummy)
            assert piece.get_position() == (i,j)
            assert piece.team is None


# --- Tests for GameEngine basic functionality ---

def test_gameengine_get_turn_and_move_counter():
    b = Board()
    engine = GameEngine(b)
    # Move counter starts at 0
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"
    engine.move_counter = 1
    team2, player2 = engine.get_turn()
    assert team2 == "black"


def test_gameengine_num_alph_conversion():
    engine = GameEngine(Board())
    move_dict = {
        1: ((1, 0), (2, 2), "base"),
        2: ((6, 1), (5, 3), "base")
    }
    human_readable = engine.num_alph(move_dict)
    assert human_readable[1][0] == "A2" or human_readable[1][1] == "C3"
    assert human_readable[2][0] == "B7" or human_readable[2][1] == "D6"  # Some representation


def test_gameengine_undo_move_restores_previous_state():
    board = Board()
    engine = GameEngine(board)
    initial_state = engine.Board.copy()

    start = (1,0)
    end = (3,0)
    move = (start, end, "base")

    engine.Board.update_board(move, "white")
    engine.game_states.append(engine.Board.copy())

    engine.undo_move()
    # After undo, board should equal initial_state
    assert engine.Board.equal(initial_state)


def test_gameengine_compare_states_identifies_equal_and_unequal():
    board1 = Board()
    board2 = Board()
    engine = GameEngine(board1)
    assert engine.compare_states(board1, board2) is True
    board2.board[0,0] = Dummy((0,0))
    assert engine.compare_states(board1, board2) is False
