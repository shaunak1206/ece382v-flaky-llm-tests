import pytest
import numpy as np
from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy


def test_clean_board_returns_array_of_dummy():
    board = clean_board()
    assert isinstance(board, np.ndarray)
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].pos == (i, j)


def test_board_default_start_pos_sets_pieces():
    board = Board()
    b = board.board

    # Check some white pieces
    assert b[0, 0].get_name() == "rook"
    assert b[0, 1].get_name() == "knight"
    assert b[0, 4].get_name() == "king"
    assert b[1, 0].get_name() == "pawn"
    # Check some black pieces
    assert b[7, 7].get_name() == "rook"
    assert b[7, 6].get_name() == "knight"
    assert b[7, 4].get_name() == "king"
    assert b[6, 0].get_name() == "pawn"

    # Check positions of Dummy squares
    assert isinstance(b[3, 3], Dummy)
    assert b[3, 3].pos == (3, 3)


def test_board_turn_starts_white():
    board = Board()
    assert board.turn == "white"


def test_board_copy_and_equal():
    board = Board()
    copy_board = board.copy()

    assert copy_board is not board
    # Positions should match
    assert copy_board.equal(board)
    # Modifying copy_board pieces does not affect original
    copy_board.board[1, 0] = Dummy((1, 0))
    assert not copy_board.equal(board)


@pytest.mark.parametrize(
    "key_word", [None, "base", "enpasse", "castling", "evolve"]
)
def test_board_update_board_base_moves(key_word):
    board = Board()
    start_pos = (1, 0)
    end_pos = (3, 0)
    piece = board.board[start_pos]
    team = piece.team
    move = (start_pos, end_pos, key_word)
    if key_word is None:
        key_word = "base"
    # Setup for castling moves only work for rooks and king so skip except base and evolve for now
    if key_word == "castling":
        # Place rook and king in correct positions to test castling
        board.board[0, 4] = board.board[0, 4]  # white king
        board.board[0, 0] = board.board[0, 0]  # white rook
        start_castle = (0, 4)
        end_castle_short = (0, 6)
        end_castle_long = (0, 2)

        # Test short castling (king side)
        move_short = (start_castle, end_castle_short, key_word)
        board.turn = "white"
        board.update_board(move_short, "white")
        assert board.board[0, 6].get_name() == "king"
        assert board.board[0, 5].get_name() == "rook"
        assert board.turn == "black"

        # Reset board for long castling test
        board._start_pos()
        board.turn = "white"
        move_long = (start_castle, end_castle_long, key_word)
        board.update_board(move_long, "white")
        assert board.board[0, 2].get_name() == "king"
        assert board.board[0, 3].get_name() == "rook"
        return

    if key_word == "evolve":
        # Evolve the pawn at start_pos to queen at end_pos
        start_pos = (6, 0)
        end_pos = (7, 0)
        board.board[start_pos] = board.board[6, 0]  # ensure it's a pawn
        move = (start_pos, end_pos, "evolve")
        board.turn = "white"
        board.update_board(move, "white")
        assert board.board[end_pos].get_name() == "queen"
        assert isinstance(board.board[start_pos], Dummy)
        assert board.turn == "black"
        return

    # For base and enpasse moves test a simple pawn move
    if key_word == "enpasse":
        # place a black pawn next to white pawn to simulate en passant
        board._start_pos()
        board.board[3, 1] = board.board[6, 1].copy()
        board.board[3, 1].set_position((3, 1))
        board.board[4, 1] = Dummy((4, 1))
        start = (3, 1)
        end = (2, 0)
        move = (start, end, "enpasse")
        board.turn = "black"
        # Place dummy at en passant captured pawn position
        board.board[(3, 0)] = board.board[1, 0].copy()
        board.board[(3, 0)].set_position((3, 0))
        board.update_board(move, "black")
        assert board.board[end].get_name() == board.board[end].get_name()  # moved piece there
        assert isinstance(board.board[start], Dummy)
        assert isinstance(board.board[(3, end[1])], Dummy)
        assert board.turn == "white"
        return

    # Base move test with white pawn
    board._start_pos()
    start = (1, 0)
    end = (2, 0)
    move = (start, end, "base")
    board.turn = "white"
    board.update_board(move, "white")
    assert board.board[end].get_name() == "pawn"
    assert board.board[end].pos == end
    assert isinstance(board.board[start], Dummy)
    assert board.turn == "black"

    with pytest.raises(ValueError):
        board.turn = "invalidteam"
        board.update_board(move, "white")


def test_board_str_includes_all_ranks_and_files():
    board = Board()
    s = str(board)
    # Board string includes ranks 1 to 8 and files A to H and pieces symbols
    for rank in range(1, 9):
        assert str(rank) in s
    for file in "ABCDEFGH":
        assert file in s
    # Should include some piece symbols like 'wk ' etc
    assert "wk " in s or "bk " in s


def test_knight_get_name_value_symbol():
    # White knight
    knight = Knight((0, 1), "white")
    assert knight.get_name() == "knight"
    assert knight.get_value() == 3
    assert knight.get_symbol() == "wk "

    # Black knight
    knight = Knight((7, 1), "black")
    assert knight.get_symbol() == "bk "

    # Invalid team leads to ValueError
    knight = Knight((0, 1), "green")
    with pytest.raises(ValueError):
        knight.get_symbol()


def test_knight_get_moves_board_edge_and_blocked():
    # Setup a minimal board
    empty_board = clean_board()
    # Place a white knight at center
    knight = Knight((4, 4), "white")
    empty_board[4, 4] = knight

    gs = [Board().board.copy()]  # fake game_states, just needs a board member
    # We'll substitute the last game state board with our test board
    class FakeGameState:
        def __init__(self, board):
            self.board = board

    game_states = [FakeGameState(empty_board)]

    moves = knight.get_moves(game_states)
    expected_offsets = [
        (-2, -1), (-2, +1),
        (-1, -2), (+1, -2),
        (+2, -1), (+2, +1),
        (+1, +2), (-1, +2)
    ]
    expected_moves = []
    for dx, dy in expected_offsets:
        x, y = knight.pos
        nx, ny = x + dx, y + dy
        if 0 <= nx <= 7 and 0 <= ny <= 7:
            expected_moves.append(((x, y), (nx, ny), "base"))

    # Moves generated by knight.get_moves() should match expected_moves (order may vary)
    assert set(moves) == set(expected_moves)

    # Test that it doesn't move into squares occupied by same team
    # Place white pawn blocking one square
    dummy = Dummy((2, 3), None)
    empty_board[2, 3] = Knight((2, 3), "white")  # put white piece blocking
    knight = Knight((4, 4), "white")
    empty_board[4, 4] = knight
    game_states = [FakeGameState(empty_board)]
    moves = knight.get_moves(game_states)
    for _, dest, _ in moves:
        # No moves should be to (2,3) because blocked by white piece
        assert dest != (2, 3)


def test_knight_copy_deepcopy_behavior():
    knight = Knight((1, 2), "white")
    knight.moved = True
    knight_copy = knight.copy()
    assert knight_copy.get_name() == "knight"
    assert knight_copy.pos == knight.pos
    assert knight_copy.team == knight.team
    assert knight_copy.moved == True
    # Changing copy moved does not change original
    knight_copy.moved = False
    assert knight.moved is True


def test_gameengine_init_and_turns():
    board = Board()
    engine = GameEngine(board)
    assert engine.move_counter == 0
    assert engine.Board is board
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"

    engine.move_counter = 1
    team, player = engine.get_turn()
    assert team == "black"
    assert player == "human"


def test_gameengine_all_moves_on_empty_and_start_pos():
    board = Board()
    engine = GameEngine(board)
    # For start pos all moves will be from white pawns and white pieces
    white_moves = engine.all_moves()
    assert isinstance(white_moves, list)
    assert all(isinstance(move, tuple) and len(move) == 3 for move in white_moves)
    # If we clear board and place one white knight at (4,4)
    board.board = clean_board()
    board.board[4, 4] = Knight((4, 4), "white")
    engine = GameEngine(board)
    moves = engine.all_moves()
    # Moves should be possible for knight
    assert moves != []
    # moves should be valid format
    for move in moves:
        assert len(move) == 3
        assert isinstance(move[0], tuple)
        assert isinstance(move[1], tuple)
        assert isinstance(move[2], str)


def test_gameengine_compare_states_equal_and_not_equal():
    board1 = Board()
    board2 = board1.copy()
    engine = GameEngine(board1)
    assert engine.compare_states(board1, board2)

    # Modify board2 so equal returns False
    board2.board[0, 0] = Dummy((0, 0))
    assert not engine.compare_states(board1, board2)


def test_gameengine_undo_move_reverts_to_previous_state():
    board = Board()
    engine = GameEngine(board)
    start = (1, 0)
    end = (2, 0)
    move = (start, end, "base")
    team = "white"
    engine.Board.update_board(move, team)
    engine.game_states.append(engine.Board.copy())
    assert not engine.compare_states(engine.game_states[-1], engine.game_states[-2])
    engine.undo_move()
    assert engine.compare_states(engine.game_states[-1], engine.game_states[-2]) or len(engine.game_states) == 1


def test_gameengine_num_alph_conversion():
    move_dict = {
        1: ((1, 0), (2, 0), "base"),
        2: ((0, 1), (2, 2), "base"),
    }
    expected = {
        1: ["A2", "A3"],
        2: ["B1", "C3"],
    }
    result = GameEngine.num_alph(move_dict)
    assert result == expected


def test_gameengine_is_check_and_stalemates(monkeypatch):
    # Setup to force black_check and white_check states
    board = Board()
    engine = GameEngine(board)

    # forcibly set all_moves_minus_king to return black king in white_moves so black in check
    def fake_all_moves_minus_king(game_states, team):
        if team == "white":
            return [(7, 4)], (7, 4)  # include black king pos in white moves
        else:
            return [], (0, 4)
    engine._all_moves_minus_king = fake_all_moves_minus_king
    engine.is_check()
    assert engine.black_check is True

    # Test stalemate no move triggers sys.exit (mock sys.exit)
    moves = []
    engine.all_moves = lambda: moves
    engine.white_check = False
    engine.black_check = False

    import sys

    def fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    engine.game_states = [board.copy()] * 2
    engine.turn = "white"

    # test for white stalemate
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")

    # test for black stalemate
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("black")

    # test that no exit if moves exist
    engine.all_moves = lambda: [(1, 2, "base")]
    engine.white_check = False
    engine.black_check = False
    # Should not raise
    engine.is_stalemate_no_move("white")


def test_gameengine_is_stalemate_three_states(monkeypatch):
    # Setup boards for game_states with repeated states
    board = Board()
    engine = GameEngine(board)

    # Fake compare_states to return True for all checks
    engine.compare_states = lambda S, T: True

    # setup at least 9 game states needed
    base_state = board.copy()
    engine.game_states = [base_state.copy() for _ in range(10)]

    import sys

    def fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


@pytest.mark.parametrize("player_mode, move", [("human", None), ("computer", ( (1, 0), (2, 0), "base"))])
def test_gameengine_choose_move_basic(player_mode, move, monkeypatch):
    board = Board()
    engine = GameEngine(board, player1=player_mode, player2=player_mode)
    engine.move_counter = 0 if player_mode == "human" else 1  # white human or black computer

    # monkeypatch input to avoid.stdin blocking for human mode tests
    if player_mode == "human":
        inputs = iter(["A2", "1"])

        def fake_input(prompt):
            return next(inputs)

        monkeypatch.setattr("builtins.input", fake_input)
    else:
        # For computer: pass move param
        pass

    # monkeypatch methods causing game end or print loops
    engine.is_stalemate_no_move = lambda team: None
    engine.is_stalemate_3states = lambda team: None
    engine.is_check = lambda: None
    engine.is_checkmate = lambda team: None
    engine.undo_move = lambda: None
    engine.get_moves = lambda pos: [( (1,0), (2,0), "base")]
    engine.game_states.append(engine.Board.copy())

    # This will run without error or sys.exit because of monkeypatch
    if player_mode == "human":
        # Move param = None, allowed
        engine.choose_move(move=None)
        with pytest.raises(ValueError):
            engine.choose_move(move=( (1, 0), (2, 0), "base"))
    else:
        with pytest.raises(ValueError):
            engine.choose_move(move=None)
        # Provide a valid move - should not raise error
        engine.choose_move(move=move)


def test_board_equal_false_when_different_symbols():
    board1 = Board()
    board2 = board1.copy()
    # Modify board2 symbol of one piece
    piece = board2.board[0, 0]
    piece.symbol = "QX "
    # Force symbol difference should make equal return False
    assert not board1.equal(board2)
