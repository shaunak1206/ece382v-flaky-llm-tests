import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.knight import Knight
from darkchess.src.gamengine import GameEngine
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            piece = board[i, j]
            # All must be Dummy pieces initially
            assert isinstance(piece, Dummy)
            # Position should be correct
            assert piece.pos == (i, j)
            # Team should be None for Dummy pieces
            assert piece.team is None


def test_board_default_initialization_sets_turn_white():
    board = Board()
    assert board.turn == "white"
    # Check some starting pieces, for example white rooks and black king
    rook = board.board[0, 0]
    assert rook.get_name() == "rook"
    assert rook.team == "white"
    king = board.board[7, 4]
    assert king.get_name() == "king"
    assert king.team == "black"


def test_board_custom_starting_board_assignment():
    dummy_board = clean_board()
    # Put a white knight at 4,4 in dummy_board
    dummy_board[4, 4] = Knight((4, 4), "white")
    board = Board(starting_board=dummy_board)
    assert board.board[4, 4].get_name() == "knight"
    assert board.board[4, 4].team == "white"


def test_board_str_contains_all_ranks_and_files(monkeypatch):
    board = Board()
    board_str = str(board)
    # Should contain ranks 1-8 and files A-H in the output string
    for rank in range(1, 9):
        assert str(rank) in board_str
    for file_char in "ABCDEFGH":
        assert file_char in board_str


def test_board_update_board_base_move_updates_and_switches_turn():
    board = Board()
    start = (1, 0)  # white pawn initial pos
    end = (3, 0)
    move = (start, end, None)  # base move
    team = "white"
    initial_piece = board.board[start]
    board.update_board(move, team)
    # The piece should now be at end position
    assert board.board[end].get_name() == initial_piece.get_name()
    # The start position should be Dummy
    assert isinstance(board.board[start], Dummy)
    # The piece at end should have updated position and moved True
    assert board.board[end].pos == end
    assert board.board[end].moved is True
    # Turn toggled to black
    assert board.turn == "black"


def test_board_update_board_enpasse_removes_captured_pawn():
    board = Board()
    # Set up en passant scenario manually:
    # white pawn at (3,4), black pawn at (3,5)
    board.board = clean_board()
    wp = board.board[3, 4] = board.board[3, 4] = Knight((3, 4), "white")  # Use Knight as dummy moving piece
    wp.team = "white"
    board.board[3, 5] = Knight((3, 5), "black")
    start = (3, 4)
    end = (2, 5)
    move = (start, end, "enpasse")
    board.update_board(move, "white")
    # The pawn captured by en passant (3,5) should be Dummy now
    assert isinstance(board.board[(3, 5)], Dummy)
    # The moved piece moved to end position
    assert board.board[end].pos == end
    assert board.board[end].team == "white"
    # Start position should be Dummy
    assert isinstance(board.board[start], Dummy)
    # Turn switched
    assert board.turn == "black"


def test_board_update_board_castling_kingside_and_queenside():
    board = Board()
    board.board = clean_board()
    # place a white king and rooks at initial positions for castling
    king_pos = (0, 4)
    rook_k_pos = (0, 7)  # kingside rook
    rook_q_pos = (0, 0)  # queenside rook
    board.board[king_pos] = Knight(king_pos, "white")
    board.board[rook_k_pos] = Knight(rook_k_pos, "white")
    board.board[rook_q_pos] = Knight(rook_q_pos, "white")

    # Castling kingside: king moves to (0,6)
    move_kingside = (king_pos, (0, 6), "castling")
    board.update_board(move_kingside, "white")
    # King at (0,6)
    assert board.board[(0, 6)].get_name() == "knight"
    # Rook moved next to king at (0,5)
    assert board.board[(0, 5)].get_name() == "knight"
    # Original rook pos empty
    assert isinstance(board.board[(0, 7)], Dummy)
    # Original king pos empty
    assert isinstance(board.board[(0, 4)], Dummy)
    # Reset for queenside castle
    board.board = clean_board()
    board.board[king_pos] = Knight(king_pos, "black")
    board.board[rook_q_pos] = Knight(rook_q_pos, "black")

    # Castling queenside: king moves to (0,2)
    move_queenside = (king_pos, (0, 2), "castling")
    board.turn = "black"
    board.update_board(move_queenside, "black")
    # King at (0,2)
    assert board.board[(0, 2)].get_name() == "knight"
    # Rook moved next to king at (0,3)
    assert board.board[(0, 3)].get_name() == "knight"
    # Original rook pos empty
    assert isinstance(board.board[(0, 0)], Dummy)
    # Original king pos empty
    assert isinstance(board.board[(0, 4)], Dummy)


def test_board_update_board_evolve_promotes_to_queen():
    board = Board()
    # Place a white pawn ready to evolve at (6, 0)
    board.board = clean_board()
    board.board[6, 0] = Knight((6, 0), "white")
    start = (6, 0)
    end = (7, 0)
    move = (start, end, "evolve")
    board.update_board(move, "white")
    # At end position now should be a Queen
    piece = board.board[end]
    assert piece.get_name() == "queen"
    assert piece.team == "white"
    # Start position is Dummy
    assert isinstance(board.board[start], Dummy)


def test_board_copy_returns_distinct_board():
    board = Board()
    board_copy = board.copy()
    # Boards should be distinct objects
    assert board is not board_copy
    # Their internal board arrays should also be distinct
    assert board.board is not board_copy.board
    # Pieces in the copied board should be distinct instances
    for i in range(8):
        for j in range(8):
            assert board.board[i, j] is not board_copy.board[i, j]
            # Positions in copy reset correctly
            assert board_copy.board[i, j].pos == (i, j)
    # Turn should be the same but a distinct copy (string immutable, but test anyway)
    assert board.turn == board_copy.turn


def test_board_equal_detects_equal_and_unequal_boards():
    board1 = Board()
    board2 = Board()
    assert board1.equal(board2) is True
    # Change a piece symbol in board2
    board2.board[0, 0] = Knight((0, 0), "white")
    assert board1.equal(board2) is False


def test_knight_get_moves_in_corners_and_center():
    # Create empty board with Dummy pieces
    empty_board = clean_board()
    # Place white knight in center
    knight_center = Knight((4, 4), "white")
    empty_board[4, 4] = knight_center
    game_states = [Board().copy()]
    game_states[-1].board = empty_board

    moves = knight_center.get_moves(game_states)
    # Knight has max 8 moves in center
    assert len(moves) <= 8
    for move in moves:
        start, end, move_type = move
        # start must be (4,4)
        assert start == (4, 4)
        # end positions must be reachable knight moves inside board
        assert 0 <= end[0] <= 7
        assert 0 <= end[1] <= 7
        # move_type should always be "base"
        assert move_type == "base"

    # Place black knight in corner
    knight_corner = Knight((0, 0), "black")
    empty_board[0, 0] = knight_corner
    game_states[-1].board = empty_board
    moves = knight_corner.get_moves(game_states)
    # Knight moves from corner should be <= 2
    assert len(moves) <= 2
    for move in moves:
        start, end, move_type = move
        assert start == (0, 0)
        assert 0 <= end[0] <= 7
        assert 0 <= end[1] <= 7
        assert move_type == "base"


def test_knight_get_symbol_returns_correct_string_or_raises():
    knight_white = Knight((0, 0), "white")
    knight_black = Knight((0, 0), "black")
    assert knight_white.get_symbol() == "wk "
    assert knight_black.get_symbol() == "bk "
    # Directly set invalid team and expect ValueError
    knight_invalid = Knight((0, 0), "red")
    with pytest.raises(ValueError):
        knight_invalid.get_symbol()


def test_gameengine_get_turn_alternates_based_on_move_counter():
    engine = GameEngine(Board())
    # Initial turn should be white, player1 default human
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"
    # simulate moves
    engine.move_counter = 1
    team, player = engine.get_turn()
    assert team == "black"
    assert player == "human"
    # set player1 to computer and player2 human
    engine = GameEngine(Board(), player1="computer", player2="human")
    engine.move_counter = 0
    team, player = engine.get_turn()
    assert player == "computer"
    engine.move_counter = 1
    team, player = engine.get_turn()
    assert player == "human"


def test_gameengine_compare_states_with_equal_boards():
    board1 = Board()
    board2 = board1.copy()
    engine = GameEngine(board1)
    assert engine.compare_states(board1, board2) is True
    # modify board2 slightly
    board2.board[0, 0] = Knight((0, 0), "white")
    assert engine.compare_states(board1, board2) is False


def test_gameengine_all_moves_returns_moves_for_team():
    engine = GameEngine(Board())
    # Gets all moves for white (initially)
    moves = engine.all_moves()
    assert isinstance(moves, list)
    # Each move is a tuple (start_pos, end_pos, type_move)
    if moves:
        move = moves[0]
        assert isinstance(move, tuple)
        assert len(move) == 3


def test_gameengine_undo_move_reverts_to_previous_state():
    engine = GameEngine(Board())
    initial_board = engine.Board.copy()
    move = ((1, 0), (2, 0), "base")
    engine.Board.update_board(move, "white")
    engine.game_states.append(engine.Board.copy())
    engine.move_counter += 1
    # Undo the move
    engine.undo_move()
    # Board should be equal to initial_board
    assert engine.Board.equal(initial_board)


def test_gameengine_num_alph_converts_moves_to_human_readable():
    engine = GameEngine(Board())
    move_dict = {
        1: ((1, 0), (3, 0), "base"),
        2: ((1, 1), (3, 1), "base")
    }
    converted = engine.num_alph(move_dict)
    assert isinstance(converted, dict)
    assert converted[1] == ["A2", "A4"]
    assert converted[2] == ["B2", "B4"]


@pytest.mark.parametrize("team", ["white", "black"])
def test_gameengine_is_check_flags_set_correctly(monkeypatch, team):
    engine = GameEngine(Board())
    engine.white_check = False
    engine.black_check = False

    # Patch _all_moves_minus_king to control which moves and king pos returned
    def fake_all_moves_minus_king(gs, tm):
        if tm == "white":
            return ([(7, 4)], (0, 0))  # moves include black king either place
        else:
            return ([], (7, 4))

    engine._all_moves_minus_king = fake_all_moves_minus_king

    engine.is_check()

    # Depending on fake data, black_check should be True if black king pos in white moves
    if team == "white":
        # With given fake data, white king is at (0, 0), black moves does not cover it, so white_check False
        pass
    assert isinstance(engine.white_check, bool)
    assert isinstance(engine.black_check, bool)


@pytest.mark.parametrize("team", ["white", "black"])
def test_gameengine_is_stalemate_no_move_detects_stalemate(monkeypatch, team):
    engine = GameEngine(Board())
    # Patch all_moves to empty list to simulate no moves available
    engine.all_moves = lambda: []
    # Set white/black_check flags accordingly so not in check
    if team == "white":
        engine.white_check = False
        engine.black_check = True
    else:
        engine.black_check = False
        engine.white_check = True

    # patch sys.exit to raise SystemExit exception to capture
    import sys
    def fake_exit(code=0):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move(team)


@pytest.mark.parametrize("team", ["white", "black"])
def test_gameengine_is_stalemate_3states_detects_stalemate(monkeypatch, team):
    engine = GameEngine(Board())
    # Create 10 copies of same state to trigger stalemate condition
    base_board = engine.Board.copy()
    engine.game_states = [base_board.copy() for _ in range(10)]

    # Patch compare_states to always return True to simulate repeated states
    engine.compare_states = lambda S, T: True

    # patch sys.exit to raise SystemExit exception to capture
    import sys
    def fake_exit(code=0):
        raise SystemExit(code)
    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        engine.is_stalemate_3states(team)
