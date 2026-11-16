import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.rook import Rook
from darkchess.src.king import King
from darkchess.src.queen import Queen
from darkchess.src.dummy_piece import Dummy
from darkchess.src.piece import Piece
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


def place(board, piece):
    board.board[piece.get_position()] = piece
    return piece


def targets(moves):
    return {m[1] for m in moves}


# ---------------- clean_board ----------------

def test_clean_board_shape_and_dummies():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None


def test_clean_board_returns_fresh_array_each_call():
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


def test_knight_symbol_invalid_team_raises():
    k = Knight((0, 0), "white")
    k.team = "green"
    with pytest.raises(ValueError):
        k.get_symbol()


def test_knight_center_has_eight_moves():
    board = empty_board()
    k = place(board, Knight((3, 3), "white"))
    moves = k.get_moves([board])
    assert len(moves) == 8
    assert targets(moves) == {(1, 2), (1, 4), (2, 1), (4, 1),
                              (5, 2), (5, 4), (2, 5), (4, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves():
    board = empty_board()
    k = place(board, Knight((0, 0), "white"))
    assert targets(k.get_moves([board])) == {(2, 1), (1, 2)}
    board2 = empty_board()
    k2 = place(board2, Knight((7, 7), "black"))
    assert targets(k2.get_moves([board2])) == {(5, 6), (6, 5)}


def test_knight_blocked_by_own_and_can_capture_enemy():
    board = empty_board()
    k = place(board, Knight((3, 3), "white"))
    place(board, Pawn((5, 4), "white"))   # own piece blocks
    place(board, Pawn((5, 2), "black"))   # enemy can be captured
    t = targets(k.get_moves([board]))
    assert (5, 4) not in t
    assert (5, 2) in t
    assert len(t) == 7


def test_black_knight_moves_respect_team():
    board = empty_board()
    k = place(board, Knight((4, 4), "black"))
    place(board, Pawn((6, 5), "black"))
    place(board, Pawn((6, 3), "white"))
    t = targets(k.get_moves([board]))
    assert (6, 5) not in t
    assert (6, 3) in t


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
    assert b.board[0, 1].team == "white"
    assert isinstance(b.board[7, 6], Knight)
    assert b.board[7, 6].team == "black"
    assert isinstance(b.board[0, 3], Queen)
    assert isinstance(b.board[7, 4], King)
    for j in range(8):
        assert isinstance(b.board[1, j], Pawn)
        assert isinstance(b.board[6, j], Pawn)
    for i in range(2, 6):
        for j in range(8):
            assert isinstance(b.board[i, j], Dummy)


def test_board_custom_starting_board():
    b = empty_board()
    assert all(isinstance(p, Dummy) for p in b.board.flatten())
    assert b.turn == "white"


def test_board_str_contains_labels_and_symbols():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    assert "wk " in s
    assert "bk " in s
    assert "8 " in s and "1 " in s


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


def test_update_board_enpasse_removes_captured_pawn():
    b = empty_board()
    place(b, Pawn((4, 3), "white"))
    place(b, Pawn((4, 4), "black"))
    b.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert isinstance(b.board[5, 4], Pawn)
    assert b.board[5, 4].team == "white"
    assert isinstance(b.board[4, 3], Dummy)
    assert isinstance(b.board[4, 4], Dummy)


def test_update_board_castling_moves_rook():
    b = empty_board()
    place(b, King((0, 4), "white"))
    place(b, Rook((0, 7), "white"))
    place(b, Rook((0, 0), "white"))
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert isinstance(b.board[0, 7], Dummy)
    assert isinstance(b.board[0, 4], Dummy)

    b2 = empty_board()
    place(b2, King((0, 4), "white"))
    place(b2, Rook((0, 0), "white"))
    b2.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(b2.board[0, 2], King)
    assert isinstance(b2.board[0, 3], Rook)
    assert isinstance(b2.board[0, 0], Dummy)


def test_update_board_evolve_creates_queen():
    b = empty_board()
    place(b, Pawn((6, 0), "white"))
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


def test_board_copy_and_equal():
    b = Board()
    c = b.copy()
    assert c is not b
    assert c.board is not b.board
    assert c.turn == b.turn
    assert b.equal(c)
    assert c.board[0, 1] is not b.board[0, 1]
    c.update_board(((0, 1), (2, 2), "base"), "white")
    assert not b.equal(c)
    # original untouched
    assert isinstance(b.board[0, 1], Knight)
    assert b.turn == "white"


# ---------------- GameEngine ----------------

def test_engine_init_and_get_turn():
    e = GameEngine(Board())
    assert e.player1 == "human" and e.player2 == "human"
    assert e.move_counter == 0
    assert len(e.game_states) == 1
    assert e.white_check is False and e.black_check is False
    assert e.get_turn() == ("white", "human")
    e.move_counter = 1
    e.player2 = "computer"
    assert e.get_turn() == ("black", "computer")


def test_engine_num_alph():
    d = {1: ((0, 1), (2, 2), "base"), 2: ((7, 7), (0, 0), "base")}
    assert GameEngine.num_alph(d) == {1: ["B1", "C3"], 2: ["H8", "A1"]}


def test_engine_get_moves_knight():
    e = GameEngine(Board())
    moves = e.get_moves((0, 1))
    assert set(moves) == {((0, 1), (2, 0), "base"), ((0, 1), (2, 2), "base")}


def test_engine_choose_move_computer_and_undo():
    e = GameEngine(Board(), "computer", "computer")
    e.choose_move(((0, 1), (2, 2), "base"))
    assert e.move_counter == 1
    assert len(e.game_states) == 2
    assert isinstance(e.Board.board[2, 2], Knight)
    assert e.get_turn()[0] == "black"
    e.undo_move()
    assert len(e.game_states) == 1
    assert isinstance(e.Board.board[0, 1], Knight)
    assert isinstance(e.Board.board[2, 2], Dummy)


def test_engine_choose_move_errors():
    e = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        e.choose_move(None)
    h = GameEngine(Board())
    with pytest.raises(ValueError):
        h.choose_move(((0, 1), (2, 2), "base"))
    bad = GameEngine(Board(), "alien", "human")
    with pytest.raises(ValueError):
        bad.choose_move()


def test_engine_is_check_detects_knight_check():
    b = empty_board()
    place(b, King((0, 4), "white"))
    place(b, King((7, 4), "black"))
    place(b, Knight((5, 3), "white"))
    e = GameEngine(b)
    e.is_check()
    assert e.black_check is True
    assert e.white_check is False


def test_engine_is_check_false_at_start():
    e = GameEngine(Board())
    e.is_check()
    assert e.black_check is False
    assert e.white_check is False


def test_engine_stalemate_3states_exits_on_repeats():
    e = GameEngine(Board())
    e.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        e.is_stalemate_3states("white")


def test_engine_stalemate_3states_not_triggered_with_few_states():
    e = GameEngine(Board())
    e.game_states = [Board() for _ in range(8)]
    e.is_stalemate_3states("white")  # should not raise
    assert e.compare_states(Board(), Board()) is True
