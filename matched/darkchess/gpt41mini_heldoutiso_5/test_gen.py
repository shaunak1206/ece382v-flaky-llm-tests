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
            # position should match coordinates
            assert board_array[i, j].pos == (i, j)
            # Dummy team should be None
            assert board_array[i, j].team is None


def test_board_default_initialization_sets_correct_pieces():
    board = Board()
    # Pawns on 2nd and 7th ranks
    for col in range(8):
        assert board.board[1, col].get_name() == "pawn"
        assert board.board[1, col].team == "white"
        assert board.board[6, col].get_name() == "pawn"
        assert board.board[6, col].team == "black"

    # Rooks
    assert board.board[0, 0].get_name() == "rook"
    assert board.board[0, 0].team == "white"
    assert board.board[0, 7].get_name() == "rook"
    assert board.board[0, 7].team == "white"
    assert board.board[7, 0].get_name() == "rook"
    assert board.board[7, 0].team == "black"
    assert board.board[7, 7].get_name() == "rook"
    assert board.board[7, 7].team == "black"

    # Knights
    assert board.board[0, 1].get_name() == "knight"
    assert board.board[0, 1].team == "white"
    assert board.board[0, 6].get_name() == "knight"
    assert board.board[0, 6].team == "white"
    assert board.board[7, 1].get_name() == "knight"
    assert board.board[7, 1].team == "black"
    assert board.board[7, 6].get_name() == "knight"
    assert board.board[7, 6].team == "black"

    # Bishops
    assert board.board[0, 2].get_name() == "bishop"
    assert board.board[0, 2].team == "white"
    assert board.board[0, 5].get_name() == "bishop"
    assert board.board[0, 5].team == "white"
    assert board.board[7, 2].get_name() == "bishop"
    assert board.board[7, 2].team == "black"
    assert board.board[7, 5].get_name() == "bishop"
    assert board.board[7, 5].team == "black"

    # Queens
    assert board.board[0, 3].get_name() == "queen"
    assert board.board[0, 3].team == "white"
    assert board.board[7, 3].get_name() == "queen"
    assert board.board[7, 3].team == "black"

    # Kings
    assert board.board[0, 4].get_name() == "king"
    assert board.board[0, 4].team == "white"
    assert board.board[7, 4].get_name() == "king"
    assert board.board[7, 4].team == "black"


def test_board_string_representation_contains_expected_rows_and_columns():
    board = Board()
    board_str = str(board)
    # Check header and footer rows with columns labels
    assert "A" in board_str
    assert "H" in board_str
    # Check presence of row numbers 1 through 8
    for n in range(1, 9):
        assert str(n) in board_str
    # Check representation contains some piece symbols
    # E.g. 'wk ' (white knight) symbol exists
    assert "wk " in board_str or "wK " in board_str


def test_board_update_base_move_updates_positions_and_turn():
    board = Board()
    start_pos = (1, 0)
    end_pos = (2, 0)
    piece_before = board.board[start_pos]
    move = (start_pos, end_pos, "base")
    current_turn = board.turn
    board.update_board(move, piece_before.team)
    # Check piece moved to new position
    piece_after = board.board[end_pos]
    assert piece_after.symbol == piece_before.symbol
    assert piece_after.get_position() == end_pos
    # Check old position has Dummy
    assert isinstance(board.board[start_pos], Dummy)
    # Check piece moved flag set
    assert piece_after.moved is True
    # Check turn toggled
    expected_turn = "black" if current_turn == "white" else "white"
    assert board.turn == expected_turn


def test_board_update_enpasse_removes_captured_pawn_and_moves_piece():
    board = Board()
    # Set up white pawn at (3,4), black pawn at (3,5)
    board.board[3, 4] = board.board[1, 4].copy()
    board.board[3, 4].set_position((3, 4))
    board.board[1, 4] = Dummy((1, 4))
    board.board[3, 5] = board.board[6, 5].copy()
    board.board[3, 5].set_position((3, 5))
    board.board[6, 5] = Dummy((6, 5))
    board.turn = "white"

    start_pos = (3, 4)
    end_pos = (2, 5)
    move = (start_pos, end_pos, "enpasse")
    board.turn = "white"
    board.board[start_pos].team = "white"
    board.board[(3, 5)].team = "black"

    board.update_board(move, "white")

    # check mover moved correctly
    assert board.board[end_pos].team == "white"
    assert isinstance(board.board[start_pos], Dummy)

    # check captured pawn removed (position on start row, end col)
    assert isinstance(board.board[(3, 5)], Dummy)
    assert board.turn == "black"


def test_board_update_castling_moves_king_and_rook_correctly():
    board = Board()
    # Castling kingside for white
    board.board[0, 4].moved = False
    board.board[0, 7].moved = False
    start_pos = (0, 4)
    # Move king to (0, 6)
    move = (start_pos, (0, 6), "castling")
    board.turn = "white"
    board.update_board(move, "white")

    # King moved
    assert board.board[0, 6].get_name() == "king"
    assert board.board[0, 6].moved is True
    # Old king pos is Dummy
    assert isinstance(board.board[0, 4], Dummy)
    # Rook moved from (0,7) to (0,5)
    assert board.board[0, 5].get_name() == "rook"
    assert board.board[0, 5].moved is True
    # Old rook pos Dummy
    assert isinstance(board.board[0, 7], Dummy)
    assert board.turn == "black"

    # Castling queenside for black
    board = Board()  # reset board
    board.board[7, 4].moved = False
    board.board[7, 0].moved = False
    start_pos = (7, 4)
    move = (start_pos, (7, 2), "castling")
    board.turn = "black"
    board.update_board(move, "black")

    # King moved
    assert board.board[7, 2].get_name() == "king"
    assert board.board[7, 2].moved is True
    # Old king pos dummy
    assert isinstance(board.board[7, 4], Dummy)
    # Rook moved from (7,0) to (7,3)
    assert board.board[7, 3].get_name() == "rook"
    assert board.board[7, 3].moved is True
    # Old rook pos dummy
    assert isinstance(board.board[7, 0], Dummy)
    assert board.turn == "white"


def test_board_update_evolve_replaces_piece_with_queen():
    board = Board()
    # place a white pawn at start pos
    start_pos = (1, 0)
    end_pos = (0, 0)
    board.board[start_pos] = board.board[1, 0].copy()
    board.board[start_pos].team = "white"
    board.turn = "white"
    move = (start_pos, end_pos, "evolve")

    board.update_board(move, "white")

    # Check that piece at end_pos is Queen and team is white
    assert board.board[end_pos].get_name() == "queen"
    assert board.board[end_pos].team == "white"
    # Check old position Dummy
    assert isinstance(board.board[start_pos], Dummy)
    # Turn changed
    assert board.turn == "black"


def test_board_copy_and_equal_methods():
    board = Board()
    board_copy = board.copy()
    # The copied board should be equal by equal()
    assert board.equal(board_copy) is True

    # Modify copied board and check inequality
    board_copy.board[1, 0] = board_copy.board[1, 0].copy()
    board_copy.board[1, 0].team = "black" if board_copy.board[1, 0].team == "white" else "white"
    assert board.equal(board_copy) is False


def test_knight_get_name_value_symbol():
    knight_white = Knight((0, 1), "white")
    knight_black = Knight((7, 6), "black")

    assert knight_white.get_name() == "knight"
    assert knight_white.get_value() == 3
    assert knight_white.get_symbol() == "wk "

    assert knight_black.get_name() == "knight"
    assert knight_black.get_value() == 3
    assert knight_black.get_symbol() == "bk "


def test_knight_copy_creates_independent_piece():
    knight = Knight((0, 1), "white")
    knight.moved = True
    copied_knight = knight.copy()
    assert isinstance(copied_knight, Knight)
    assert copied_knight.get_position() == knight.get_position()
    assert copied_knight.team == knight.team
    assert copied_knight.moved == knight.moved

    # Changing copied knight should not affect original
    copied_knight.moved = False
    assert knight.moved is True


def test_knight_get_moves_from_center_empty_board():
    board = Board()
    # Clear board, put only one white knight at center
    empty_board = clean_board()
    knight_pos = (4, 4)
    white_knight = Knight(knight_pos, "white")
    empty_board[knight_pos] = white_knight
    board.board = empty_board
    game_states = [board.copy()]

    moves = white_knight.get_moves(game_states)
    expected_moves = {
        (4, 4), (2, 3), (2, 5), (3, 2), (5, 2), (6, 3), (6, 5), (5, 6), (3, 6)
    }

    move_ends = {move[1] for move in moves}
    assert all(0 <= pos[0] < 8 and 0 <= pos[1] < 8 for pos in move_ends)
    # Knight should have 8 possible moves from center (if none blocked by same team)
    assert len(moves) == 8
    assert (4,4, 'base') not in moves  # should never move to itself


def test_knight_get_moves_blocked_by_same_team_piece():
    board = Board()
    empty_board = clean_board()
    knight_pos = (4, 4)
    white_knight = Knight(knight_pos, "white")
    empty_board[knight_pos] = white_knight
    # Put white piece in one knight move position (e.g., (6,5))
    blocking_piece = Knight((6, 5), "white")
    empty_board[6, 5] = blocking_piece
    board.board = empty_board
    game_states = [board.copy()]

    moves = white_knight.get_moves(game_states)
    move_ends = {move[1] for move in moves}
    # The blocked square should not be in moves
    assert (6, 5) not in move_ends


def test_gameengine_get_turn_and_move_counter():
    board = Board()
    engine = GameEngine(board)
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_gameengine_num_alph_conversion():
    engine = GameEngine(Board())
    move_dict = {
        1: ((1, 0), (2, 0), "base"),
        2: ((1, 1), (3, 2), "base")
    }
    converted = engine.num_alph(move_dict)
    assert converted[1][0] == "A2"
    assert converted[1][1] == "A3"
    assert converted[2][0] == "B2"
    assert converted[2][1] == "C4"


def test_gameengine_undo_move_restores_previous_state():
    board = Board()
    engine = GameEngine(board)
    start_pos = (1, 0)
    end_pos = (2, 0)
    move = (start_pos, end_pos, "base")
    team = board.board[start_pos].team
    engine.Board.update_board(move, team)
    engine.game_states.append(engine.Board.copy())
    prev_state = engine.game_states[-2]
    engine.undo_move()
    # After undo, board should equal previous state
    assert engine.Board.equal(prev_state)


@pytest.mark.parametrize("initial_turn,expected_turn", [("white", "black"), ("black", "white")])
def test_board_update_board_turn_alternation(initial_turn, expected_turn):
    board = Board()
    board.turn = initial_turn
    start = (1, 0)
    end = (2, 0)
    board.board[end] = board.board[start].copy()
    board.board[start] = Dummy(start)
    board.board[end].set_position(end)

    move = (start, end, None)  # None defaults to 'base'
    board.update_board(move, board.board[end].team)
    assert board.turn == expected_turn


def test_board_update_board_raises_value_error_on_invalid_key_word():
    board = Board()
    start = (1, 0)
    end = (2, 0)
    move = (start, end, "invalid_keyword")
    board.turn = "white"
    # Expect ValueError if turn is not 'white' or 'black' after update_board called with unknown keyword
    board.turn = "unknown"
    with pytest.raises(ValueError):
        board.update_board(move, "white")
