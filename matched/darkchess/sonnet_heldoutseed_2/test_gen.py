import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.dummy_piece import Dummy


def empty_board():
    return Board(starting_board=clean_board())


def ends(moves):
    return set(m[1] for m in moves)


# ---------------- clean_board ----------------

def test_clean_board_shape_and_content():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None


def test_clean_board_returns_independent_arrays():
    a = clean_board()
    b = clean_board()
    a[0, 0] = Knight((0, 0), "white")
    assert isinstance(b[0, 0], Dummy)


# ---------------- Knight ----------------

def test_knight_name_value_symbol():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    with pytest.raises(ValueError):
        k = Knight((3, 3), "green")
        k.get_symbol()


def test_knight_center_has_eight_moves():
    board = empty_board()
    k = Knight((3, 3), "white")
    board.board[3, 3] = k
    moves = k.get_moves([board])
    expected = {(1, 2), (1, 4), (2, 1), (4, 1), (5, 2), (5, 4), (4, 5), (2, 5)}
    assert len(moves) == 8
    assert ends(moves) == expected
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves():
    board = empty_board()
    k = Knight((0, 0), "black")
    board.board[0, 0] = k
    moves = k.get_moves([board])
    assert ends(moves) == {(2, 1), (1, 2)}


def test_knight_blocked_by_own_and_can_capture_enemy():
    board = empty_board()
    k = Knight((3, 3), "white")
    board.board[3, 3] = k
    board.board[5, 4] = Knight((5, 4), "white")
    assert (5, 4) not in ends(k.get_moves([board]))
    assert len(k.get_moves([board])) == 7

    board.board[5, 4] = Knight((5, 4), "black")
    assert (5, 4) in ends(k.get_moves([board]))
    assert len(k.get_moves([board])) == 8


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


# ---------------- Board ----------------

def test_board_default_setup():
    b = Board()
    assert b.turn == "white"
    assert b.board.shape == (8, 8)
    assert isinstance(b.board[0, 1], Knight)
    assert isinstance(b.board[7, 6], Knight)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[0, 3], Queen)
    assert b.board[0, 1].team == "white"
    assert b.board[7, 1].team == "black"
    pieces = [b.board[i, j] for i in range(8) for j in range(8)
              if b.board[i, j].team is not None]
    assert len(pieces) == 32
    for j in range(8):
        assert isinstance(b.board[1, j], Pawn)
        assert isinstance(b.board[6, j], Pawn)
    for i in range(2, 6):
        for j in range(8):
            assert isinstance(b.board[i, j], Dummy)


def test_board_str_contains_labels():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    assert s.count("\n") == 10
    assert "wk " in s and "bk " in s


def test_update_board_base_move():
    b = Board()
    pawn = b.board[1, 0]
    b.update_board(((1, 0), (3, 0), "base"), "white")
    assert b.board[3, 0] is pawn
    assert isinstance(b.board[1, 0], Dummy)
    assert pawn.get_position() == (3, 0)
    assert pawn.moved is True
    assert b.turn == "black"
    b.update_board(((6, 0), (4, 0), None), "black")
    assert isinstance(b.board[4, 0], Pawn)
    assert b.turn == "white"


def test_update_board_enpassant_removes_captured_pawn():
    b = Board()
    b.update_board(((1, 4), (4, 4), "base"), "white")
    b.update_board(((6, 3), (4, 3), "base"), "black")
    assert isinstance(b.board[4, 3], Pawn)
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert isinstance(b.board[4, 3], Dummy)
    assert isinstance(b.board[4, 4], Dummy)


def test_update_board_kingside_castling_moves_rook():
    b = Board()
    b.board[0, 5] = Dummy((0, 5))
    b.board[0, 6] = Dummy((0, 6))
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert isinstance(b.board[0, 7], Dummy)
    assert isinstance(b.board[0, 4], Dummy)
    assert b.board[0, 5].get_position() == (0, 5)


def test_update_board_evolve_creates_queen():
    b = empty_board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)
    assert b.turn == "black"


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.board[0, 1] is not b.board[0, 1]
    assert b.equal(c)
    assert c.turn == b.turn
    c.update_board(((1, 0), (3, 0), "base"), "white")
    assert not b.equal(c)
    assert isinstance(b.board[1, 0], Pawn)
    assert b.turn == "white"


# ---------------- GameEngine ----------------

def test_engine_initial_state():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")
    assert engine.white_check is False
    assert engine.black_check is False
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    assert ends(moves) == {(2, 0), (2, 2)}


def test_num_alph_conversion():
    result = GameEngine.num_alph({1: ((1, 0), (3, 0), "base"),
                                  2: ((0, 6), (2, 7), "base")})
    assert result == {1: ["A2", "A4"], 2: ["G1", "H3"]}


def test_compare_states():
    engine = GameEngine(Board())
    a = Board()
    b = Board()
    assert engine.compare_states(a, b)
    b.update_board(((1, 0), (3, 0), "base"), "white")
    assert not engine.compare_states(a, b)


def test_choose_move_computer_mode_updates_state():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[3, 4], Pawn)
    assert engine.get_turn()[0] == "black"


def test_choose_move_computer_requires_move():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    with pytest.raises(ValueError):
        engine.choose_move()


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    original = engine.Board.copy()
    engine.Board.update_board(((1, 0), (3, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)


def test_is_check_detects_knight_attack_on_king():
    board = empty_board()
    board.board[0, 0] = King((0, 0), "white")
    board.board[7, 7] = King((7, 7), "black")
    board.board[5, 6] = Knight((5, 6), "white")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_is_check_false_at_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.black_check is False
    assert engine.white_check is False
