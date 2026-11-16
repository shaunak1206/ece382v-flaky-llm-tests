import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.piece import Piece


def _knight_move_set(moves):
    return set((m[0], m[1], m[2]) for m in moves)


# ---------------------------- clean_board ----------------------------

def test_clean_board_shape_and_contents():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None


def test_clean_board_returns_independent_boards():
    b1 = clean_board()
    b2 = clean_board()
    b1[0, 0] = Knight((0, 0), "white")
    assert isinstance(b2[0, 0], Dummy)
    assert b1[3, 3] is not b2[3, 3]


# ------------------------------ Knight ------------------------------

def test_knight_basic_attributes():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    with pytest.raises(ValueError):
        k = Knight((0, 1), "green")
        k.get_symbol()
        k.get_moves([Board()])


def test_knight_moves_white_start_position():
    board = Board()
    knight = board.board[0, 1]
    moves = knight.get_moves([board])
    assert _knight_move_set(moves) == {
        ((0, 1), (2, 0), "base"),
        ((0, 1), (2, 2), "base"),
    }


def test_knight_moves_black_start_position():
    board = Board()
    knight = board.board[7, 1]
    moves = knight.get_moves([board])
    assert _knight_move_set(moves) == {
        ((7, 1), (5, 0), "base"),
        ((7, 1), (5, 2), "base"),
    }


def test_knight_center_empty_board_has_eight_moves():
    grid = clean_board()
    knight = Knight((3, 3), "white")
    grid[3, 3] = knight
    board = Board(starting_board=grid)
    moves = knight.get_moves([board])
    assert len(moves) == 8
    ends = set(m[1] for m in moves)
    assert ends == {(1, 2), (1, 4), (2, 1), (4, 1),
                    (5, 2), (5, 4), (2, 5), (4, 5)}


def test_knight_corner_has_two_moves():
    grid = clean_board()
    knight = Knight((0, 0), "black")
    grid[0, 0] = knight
    board = Board(starting_board=grid)
    moves = knight.get_moves([board])
    assert set(m[1] for m in moves) == {(2, 1), (1, 2)}


def test_knight_can_capture_enemy_but_not_friend():
    grid = clean_board()
    knight = Knight((3, 3), "white")
    grid[3, 3] = knight
    grid[5, 4] = Knight((5, 4), "black")  # enemy
    grid[5, 2] = Knight((5, 2), "white")  # friend
    board = Board(starting_board=grid)
    ends = set(m[1] for m in knight.get_moves([board]))
    assert (5, 4) in ends
    assert (5, 2) not in ends
    assert len(ends) == 7


def test_knight_copy_is_independent():
    k = Knight((0, 1), "white")
    k.moved = True
    c = k.copy()
    assert isinstance(c, Knight)
    assert c is not k
    assert c.team == "white"
    assert c.moved is True
    assert c.get_position() == (0, 1)
    c.set_position((2, 2))
    assert k.get_position() == (0, 1)


# ------------------------------- Board -------------------------------

def test_board_default_setup():
    board = Board()
    assert board.turn == "white"
    assert board.board.shape == (8, 8)
    assert board.board[0, 4].get_symbol() == "wK "
    assert board.board[7, 4].get_symbol() == "bK "
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[7, 6], Knight)
    for i in range(2, 6):
        for j in range(8):
            assert isinstance(board.board[i, j], Dummy)
    for j in range(8):
        assert board.board[1, j].team == "white"
        assert board.board[6, j].team == "black"
        assert board.board[0, j].team == "white"
        assert board.board[7, j].team == "black"


def test_board_custom_starting_board_used():
    grid = clean_board()
    board = Board(starting_board=grid)
    assert board.board is grid
    assert board.turn == "white"


def test_board_update_board_base_move():
    board = Board()
    board.update_board(((0, 1), (2, 2), "base"), "white")
    moved = board.board[2, 2]
    assert isinstance(moved, Knight)
    assert moved.get_position() == (2, 2)
    assert moved.moved is True
    assert isinstance(board.board[0, 1], Dummy)
    assert board.turn == "black"


def test_board_update_board_none_keyword_is_base_and_turn_toggles():
    board = Board()
    board.update_board(((0, 1), (2, 0), None), "white")
    assert isinstance(board.board[2, 0], Knight)
    assert board.turn == "black"
    board.update_board(((7, 1), (5, 0), None), "black")
    assert isinstance(board.board[5, 0], Knight)
    assert board.turn == "white"


def test_board_copy_is_deep_and_equal():
    board = Board()
    cp = board.copy()
    assert cp is not board
    assert cp.equal(board)
    assert cp.turn == board.turn
    assert cp.board[0, 1] is not board.board[0, 1]
    cp.update_board(((0, 1), (2, 2), "base"), "white")
    assert isinstance(board.board[0, 1], Knight)
    assert not cp.equal(board)


# ---------------------------- GameEngine ----------------------------

def test_engine_initial_state():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")
    assert engine.white_check is False
    assert engine.black_check is False


def test_engine_get_turn_alternates_with_players():
    engine = GameEngine(Board(), player1="computer", player2="human")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    assert _knight_move_set(moves) == {
        ((0, 1), (2, 0), "base"),
        ((0, 1), (2, 2), "base"),
    }


def test_engine_all_moves_includes_white_knight_moves():
    engine = GameEngine(Board())
    moves = engine.all_moves()
    assert all(m[0][0] in (0, 1) for m in moves)
    assert ((0, 1), (2, 2), "base") in moves
    assert ((0, 6), (2, 5), "base") in moves


def test_engine_num_alph_conversion():
    move_dict = {1: ((0, 1), (2, 2), "base"), 2: ((1, 4), (3, 4), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["B1", "C3"], 2: ["E2", "E4"]}


def test_engine_is_check_false_at_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False


def test_engine_computer_move_updates_state():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    engine.choose_move(((0, 1), (2, 2), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[2, 2], Knight)
    assert engine.get_turn()[0] == "black"


def test_engine_choose_move_invalid_inputs_raise():
    engine = GameEngine(Board(), player1="computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    engine2 = GameEngine(Board())
    with pytest.raises(ValueError):
        engine2.choose_move(((0, 1), (2, 2), "base"))
    engine3 = GameEngine(Board(), player1="alien")
    with pytest.raises(ValueError):
        engine3.choose_move(((0, 1), (2, 2), "base"))


def test_engine_undo_move_restores_previous_board():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.Board.update_board(((0, 1), (2, 2), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)
    assert isinstance(engine.Board.board[0, 1], Knight)


def test_engine_compare_states():
    engine = GameEngine(Board())
    a = Board()
    b = Board()
    assert engine.compare_states(a, b) is True
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert engine.compare_states(a, b) is False
