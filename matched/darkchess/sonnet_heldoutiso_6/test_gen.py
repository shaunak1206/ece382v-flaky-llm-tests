import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen
from darkchess.src.dummy_piece import Dummy
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


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


# ------------------------------ Knight ------------------------------

def test_knight_basic_attributes():
    k = Knight((0, 1), "white")
    assert k.get_name() == "knight"
    assert k.get_value() == 3
    assert k.get_symbol() == "wk "
    assert Knight((7, 1), "black").get_symbol() == "bk "


def test_knight_invalid_team_raises():
    with pytest.raises(ValueError):
        k = Knight((0, 1), "green")
        k.get_symbol()


def test_knight_moves_center_empty_board():
    b = empty_board()
    k = Knight((3, 3), "white")
    b.board[3, 3] = k
    moves = k.get_moves([b])
    ends = {m[1] for m in moves}
    assert ends == {(1, 2), (1, 4), (2, 1), (4, 1),
                    (5, 2), (5, 4), (4, 5), (2, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_moves_corner():
    b = empty_board()
    k = Knight((0, 0), "black")
    b.board[0, 0] = k
    ends = {m[1] for m in k.get_moves([b])}
    assert ends == {(2, 1), (1, 2)}


def test_knight_start_position_moves_blocked_by_own_pawn():
    b = Board()
    k = b.board[0, 1]
    ends = {m[1] for m in k.get_moves([b])}
    assert ends == {(2, 0), (2, 2)}


def test_knight_can_capture_enemy_but_not_friend():
    b = empty_board()
    k = Knight((3, 3), "white")
    b.board[3, 3] = k
    b.board[5, 4] = Pawn((5, 4), "black")
    b.board[5, 2] = Pawn((5, 2), "white")
    ends = {m[1] for m in k.get_moves([b])}
    assert (5, 4) in ends
    assert (5, 2) not in ends
    assert len(ends) == 7


def test_knight_black_moves_respect_black_pieces():
    b = empty_board()
    k = Knight((4, 4), "black")
    b.board[4, 4] = k
    b.board[6, 5] = Pawn((6, 5), "black")
    b.board[6, 3] = Pawn((6, 3), "white")
    ends = {m[1] for m in k.get_moves([b])}
    assert (6, 5) not in ends
    assert (6, 3) in ends


def test_knight_copy_is_independent():
    k = Knight((2, 2), "white")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "white"
    assert c.get_position() == (2, 2)
    assert c.moved is True
    c.set_position((4, 4))
    assert k.get_position() == (2, 2)


# ------------------------------- Board -------------------------------

def test_board_default_setup():
    b = Board()
    assert b.turn == "white"
    count = 0
    for i in range(8):
        for j in range(8):
            if b.board[i, j].team is not None:
                count += 1
    assert count == 32
    assert isinstance(b.board[0, 1], Knight)
    assert isinstance(b.board[7, 6], Knight)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[7, 3], Queen)
    assert isinstance(b.board[1, 3], Pawn)
    assert b.board[0, 0].team == "white"
    assert b.board[7, 0].team == "black"
    assert isinstance(b.board[3, 3], Dummy)


def test_board_custom_starting_board():
    b = empty_board()
    assert b.turn == "white"
    assert all(isinstance(b.board[i, j], Dummy)
               for i in range(8) for j in range(8))


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
    b.update_board(((7, 1), (5, 2), "base"), "black")
    assert b.turn == "white"


def test_update_board_none_keyword_acts_as_base():
    b = Board()
    b.update_board(((0, 6), (2, 5), None), "white")
    assert isinstance(b.board[2, 5], Knight)
    assert isinstance(b.board[0, 6], Dummy)


def test_update_board_enpasse_removes_captured_pawn():
    b = Board()
    b.board[4, 4] = Pawn((4, 4), "white")
    b.board[4, 3] = Pawn((4, 3), "black")
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 3], Dummy)


def test_update_board_castling_kingside():
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
    b = Board()
    b.board[6, 0] = Pawn((6, 0), "white")
    b.board[7, 0] = Dummy((7, 0))
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.equal(b)
    assert c.turn == b.turn
    c.update_board(((0, 1), (2, 2), "base"), "white")
    assert not c.equal(b)
    assert isinstance(b.board[0, 1], Knight)
    assert b.turn == "white"


# ---------------------------- GameEngine ----------------------------

def test_engine_init_and_turn():
    e = GameEngine(Board())
    assert e.move_counter == 0
    assert len(e.game_states) == 1
    assert e.player1 == "human" and e.player2 == "human"
    assert e.get_turn() == ("white", "human")
    e.move_counter = 1
    assert e.get_turn() == ("black", "human")
    e2 = GameEngine(Board(), "computer", "human")
    assert e2.get_turn() == ("white", "computer")


def test_engine_get_moves_for_knight():
    e = GameEngine(Board())
    ends = {m[1] for m in e.get_moves((0, 1))}
    assert ends == {(2, 0), (2, 2)}


def test_num_alph_conversion():
    move_dict = {1: ((0, 1), (2, 2), "base"), 2: ((0, 0), (7, 7), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["B1", "C3"], 2: ["A1", "H8"]}


def test_choose_move_computer_mode_updates_state():
    e = GameEngine(Board(), "computer", "computer")
    e.choose_move(((0, 1), (2, 2), "base"))
    assert e.move_counter == 1
    assert len(e.game_states) == 2
    assert isinstance(e.Board.board[2, 2], Knight)
    assert e.get_turn() == ("black", "computer")
    assert e.white_check is False and e.black_check is False


def test_choose_move_invalid_modes_raise():
    e = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        e.choose_move(None)
    e2 = GameEngine(Board())
    with pytest.raises(ValueError):
        e2.choose_move(((0, 1), (2, 2), "base"))


def test_undo_move_restores_previous_state():
    e = GameEngine(Board(), "computer", "computer")
    original = e.game_states[0].copy()
    e.choose_move(((0, 1), (2, 2), "base"))
    e.undo_move()
    assert len(e.game_states) == 1
    assert e.Board.equal(original)


def test_is_check_detects_knight_check():
    b = empty_board()
    b.board[0, 0] = King((0, 0), "white")
    b.board[7, 4] = King((7, 4), "black")
    b.board[5, 3] = Knight((5, 3), "white")
    e = GameEngine(b)
    e.is_check()
    assert e.black_check is True
    assert e.white_check is False


def test_is_check_false_at_start():
    e = GameEngine(Board())
    e.is_check()
    assert e.white_check is False
    assert e.black_check is False


def test_compare_states():
    e = GameEngine(Board())
    a = Board()
    c = Board()
    assert e.compare_states(a, c)
    c.update_board(((0, 1), (2, 2), "base"), "white")
    assert not e.compare_states(a, c)


def test_stalemate_3states_triggers_exit_on_repeated_states():
    e = GameEngine(Board())
    e.game_states = [e.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        e.is_stalemate_3states("white")


def test_stalemate_3states_no_exit_with_few_states():
    e = GameEngine(Board())
    e.game_states = [e.Board.copy() for _ in range(5)]
    e.is_stalemate_3states("white")
