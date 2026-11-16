import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.piece import Piece
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen


def empty_board():
    return Board(starting_board=clean_board())


def end_positions(moves):
    return {m[1] for m in moves}


# ---------------------------- clean_board ----------------------------

def test_clean_board_shape_and_content():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Piece)
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None


def test_clean_board_returns_new_array_each_call():
    a = clean_board()
    b = clean_board()
    assert a is not b
    a[0, 0] = Knight((0, 0), "white")
    assert isinstance(b[0, 0], Dummy)


# ------------------------------ Knight -------------------------------

def test_knight_basic_info():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    b = empty_board()
    with pytest.raises(ValueError):
        k = Knight((3, 3), "green")
        k.get_symbol()
        k.get_moves([b])


def test_knight_moves_center_empty_board():
    b = empty_board()
    k = Knight((4, 4), "white")
    b.board[4, 4] = k
    moves = k.get_moves([b])
    assert len(moves) == 8
    assert end_positions(moves) == {(2, 3), (2, 5), (3, 2), (5, 2),
                                    (6, 3), (6, 5), (5, 6), (3, 6)}
    for m in moves:
        assert m[0] == (4, 4)
        assert m[2] == "base"


def test_knight_moves_corner_black():
    b = empty_board()
    k = Knight((7, 7), "black")
    b.board[7, 7] = k
    moves = k.get_moves([b])
    assert end_positions(moves) == {(5, 6), (6, 5)}


def test_knight_cannot_capture_friend_but_can_capture_enemy():
    b = empty_board()
    k = Knight((0, 0), "white")
    b.board[0, 0] = k
    b.board[2, 1] = Knight((2, 1), "white")
    b.board[1, 2] = Knight((1, 2), "black")
    moves = k.get_moves([b])
    assert end_positions(moves) == {(1, 2)}


def test_knight_moves_from_starting_position():
    b = Board()
    k = b.board[0, 1]
    assert isinstance(k, Knight)
    assert end_positions(k.get_moves([b])) == {(2, 0), (2, 2)}


def test_knight_copy_is_independent():
    k = Knight((0, 1), "white")
    k.moved = True
    c = k.copy()
    assert isinstance(c, Knight)
    assert c is not k
    assert c.team == "white"
    assert c.moved is True
    c.set_position((2, 2))
    assert k.get_position() == (0, 1)


# ------------------------------- Board -------------------------------

def test_board_default_setup():
    b = Board()
    assert b.turn == "white"
    assert isinstance(b.board[0, 1], Knight)
    assert b.board[0, 1].team == "white"
    assert isinstance(b.board[7, 6], Knight)
    assert b.board[7, 6].team == "black"
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[7, 3], Queen)
    assert isinstance(b.board[1, 3], Pawn)
    assert isinstance(b.board[4, 4], Dummy)


def test_board_custom_starting_board():
    raw = clean_board()
    raw[3, 3] = Knight((3, 3), "white")
    b = Board(starting_board=raw)
    assert isinstance(b.board[3, 3], Knight)
    assert isinstance(b.board[0, 0], Dummy)
    assert b.turn == "white"


def test_board_str_contains_headers_and_rows():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    assert "1 " in s and "8 " in s
    assert s.count("\n") == 10


def test_update_board_base_move():
    b = Board()
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert isinstance(b.board[2, 2], Knight)
    assert b.board[2, 2].get_position() == (2, 2)
    assert b.board[2, 2].moved is True
    assert isinstance(b.board[0, 1], Dummy)
    assert b.turn == "black"
    b.update_board(((7, 1), (5, 2), None), "black")
    assert isinstance(b.board[5, 2], Knight)
    assert b.turn == "white"


def test_update_board_castling_kingside():
    b = Board()
    b.board[0, 5] = Dummy((0, 5))
    b.board[0, 6] = Dummy((0, 6))
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert isinstance(b.board[0, 4], Dummy)
    assert isinstance(b.board[0, 7], Dummy)
    assert b.board[0, 5].get_position() == (0, 5)


def test_update_board_enpasse_removes_captured_pawn():
    b = empty_board()
    b.board[4, 4] = Pawn((4, 4), "white")
    b.board[4, 3] = Pawn((4, 3), "black")
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert isinstance(b.board[4, 3], Dummy)
    assert isinstance(b.board[4, 4], Dummy)


def test_update_board_evolve_creates_queen():
    b = empty_board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.board[0, 1] is not b.board[0, 1]
    assert b.equal(c)
    assert c.turn == b.turn
    c.update_board(((0, 1), (2, 2), "base"), "white")
    assert not b.equal(c)
    assert isinstance(b.board[0, 1], Knight)
    assert b.turn == "white"


# ---------------------------- GameEngine -----------------------------

def test_engine_initial_state():
    e = GameEngine(Board())
    assert e.move_counter == 0
    assert len(e.game_states) == 1
    assert e.get_turn() == ("white", "human")
    assert e.white_check is False
    assert e.black_check is False
    e.move_counter = 1
    assert e.get_turn() == ("black", "human")
    e2 = GameEngine(Board(), "computer", "human")
    assert e2.get_turn() == ("white", "computer")


def test_engine_num_alph():
    result = GameEngine.num_alph({1: ((0, 1), (2, 2), "base"),
                                  2: ((6, 7), (4, 0), "base")})
    assert result == {1: ["B1", "C3"], 2: ["H7", "A5"]}


def test_engine_get_moves_for_knight():
    e = GameEngine(Board())
    moves = e.get_moves((0, 1))
    assert end_positions(moves) == {(2, 0), (2, 2)}


def test_engine_computer_move_advances_game():
    e = GameEngine(Board(), "computer", "computer")
    e.choose_move(((0, 1), (2, 2), "base"))
    assert e.move_counter == 1
    assert len(e.game_states) == 2
    assert isinstance(e.Board.board[2, 2], Knight)
    assert e.get_turn()[0] == "black"
    assert e.white_check is False and e.black_check is False


def test_engine_choose_move_errors():
    e = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        e.choose_move(None)
    h = GameEngine(Board())
    with pytest.raises(ValueError):
        h.choose_move(((0, 1), (2, 2), "base"))
    bad = GameEngine(Board(), "robot", "robot")
    with pytest.raises(ValueError):
        bad.choose_move(((0, 1), (2, 2), "base"))


def test_engine_undo_move_restores_previous_state():
    e = GameEngine(Board())
    e.Board.update_board(((0, 1), (2, 2), "base"), "white")
    e.game_states.append(e.Board.copy())
    assert len(e.game_states) == 2
    e.undo_move()
    assert len(e.game_states) == 1
    assert isinstance(e.Board.board[0, 1], Knight)
    assert isinstance(e.Board.board[2, 2], Dummy)


def test_engine_is_check_detects_rook_attack():
    raw = clean_board()
    raw[0, 4] = King((0, 4), "white")
    raw[7, 7] = King((7, 7), "black")
    raw[4, 4] = Rook((4, 4), "black")
    e = GameEngine(Board(starting_board=raw))
    e.is_check()
    assert e.white_check is True
    assert e.black_check is False


def test_engine_is_check_false_on_starting_board():
    e = GameEngine(Board())
    e.is_check()
    assert e.white_check is False
    assert e.black_check is False


def test_engine_stalemate_by_repetition_exits():
    e = GameEngine(Board())
    e.game_states = [Board().copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        e.is_stalemate_3states("white")


def test_engine_no_repetition_stalemate_with_few_states():
    e = GameEngine(Board())
    e.game_states = [Board().copy() for _ in range(8)]
    e.is_stalemate_3states("white")  # should not exit


def test_engine_compare_states():
    e = GameEngine(Board())
    a = Board()
    b = Board()
    assert e.compare_states(a, b) is True
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert e.compare_states(a, b) is False
