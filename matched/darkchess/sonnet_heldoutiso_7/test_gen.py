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


def _empty_board_engine_board():
    return Board(starting_board=clean_board())


def _knight_moves(knight, extra=()):
    b = clean_board()
    b[knight.get_position()] = knight
    for piece in extra:
        b[piece.get_position()] = piece
    board = Board(starting_board=b)
    return knight.get_moves([board])


# ---------------- clean_board ----------------

def test_clean_board_shape_and_content():
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


# ---------------- Knight ----------------

def test_knight_basic_info():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_center_has_eight_moves():
    k = Knight((4, 4), "white")
    moves = _knight_moves(k)
    ends = {m[1] for m in moves}
    assert len(moves) == 8
    assert ends == {(2, 3), (2, 5), (3, 2), (5, 2),
                    (6, 3), (6, 5), (3, 6), (5, 6)}
    assert all(m[0] == (4, 4) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves():
    k = Knight((0, 0), "black")
    moves = _knight_moves(k)
    assert {m[1] for m in moves} == {(2, 1), (1, 2)}


def test_knight_blocked_by_friend_and_captures_enemy():
    k = Knight((4, 4), "white")
    friend = Pawn((6, 5), "white")
    enemy = Pawn((6, 3), "black")
    moves = _knight_moves(k, extra=[friend, enemy])
    ends = {m[1] for m in moves}
    assert (6, 5) not in ends
    assert (6, 3) in ends
    assert len(moves) == 7


def test_knight_black_blocked_by_friend():
    k = Knight((4, 4), "black")
    friend = Pawn((2, 3), "black")
    enemy = Pawn((2, 5), "white")
    moves = _knight_moves(k, extra=[friend, enemy])
    ends = {m[1] for m in moves}
    assert (2, 3) not in ends
    assert (2, 5) in ends


def test_knight_copy_is_independent():
    k = Knight((3, 3), "white")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "white"
    assert c.moved is True
    assert c.get_position() == (3, 3)


# ---------------- Board ----------------

def test_board_default_setup():
    board = Board()
    assert board.turn == "white"
    assert board.board.shape == (8, 8)
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[7, 6], Knight)
    assert board.board[0, 1].team == "white"
    assert board.board[7, 1].team == "black"
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[4, 4], Dummy)
    for j in range(8):
        assert isinstance(board.board[1, j], Pawn)
        assert isinstance(board.board[6, j], Pawn)


def test_board_custom_starting_board():
    b = clean_board()
    b[3, 3] = Knight((3, 3), "white")
    board = Board(starting_board=b)
    assert isinstance(board.board[3, 3], Knight)
    assert isinstance(board.board[0, 0], Dummy)


def test_board_str_contains_headers():
    s = str(Board())
    assert "A  B  C  D  E  F  G  H" in s
    assert "wk " in s
    assert "bk " in s


def test_update_board_base_move():
    board = Board()
    board.update_board(((0, 1), (2, 2), "base"), "white")
    assert isinstance(board.board[2, 2], Knight)
    assert board.board[2, 2].get_position() == (2, 2)
    assert board.board[2, 2].moved is True
    assert isinstance(board.board[0, 1], Dummy)
    assert board.turn == "black"
    board.update_board(((7, 1), (5, 2), "base"), "black")
    assert board.turn == "white"


def test_update_board_none_keyword_is_base():
    board = Board()
    board.update_board(((0, 1), (2, 0), None), "white")
    assert isinstance(board.board[2, 0], Knight)
    assert isinstance(board.board[0, 1], Dummy)


def test_update_board_enpassant_removes_captured_pawn():
    b = clean_board()
    b[4, 4] = Pawn((4, 4), "white")
    b[4, 5] = Pawn((4, 5), "black")
    board = Board(starting_board=b)
    board.update_board(((4, 4), (5, 5), "enpasse"), "white")
    assert isinstance(board.board[5, 5], Pawn)
    assert board.board[5, 5].team == "white"
    assert isinstance(board.board[4, 5], Dummy)
    assert isinstance(board.board[4, 4], Dummy)


def test_update_board_castling_kingside():
    b = clean_board()
    b[0, 4] = King((0, 4), "white")
    b[0, 7] = Rook((0, 7), "white")
    board = Board(starting_board=b)
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)
    assert board.board[0, 5].get_position() == (0, 5)


def test_update_board_evolve_creates_queen():
    b = clean_board()
    b[6, 0] = Pawn((6, 0), "white")
    board = Board(starting_board=b)
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert isinstance(board.board[6, 0], Dummy)


def test_board_copy_and_equal():
    board = Board()
    c = board.copy()
    assert c is not board
    assert c.equal(board)
    assert c.turn == board.turn
    board.update_board(((0, 1), (2, 2), "base"), "white")
    assert not c.equal(board)
    assert isinstance(c.board[0, 1], Knight)
    assert c.turn == "white"


# ---------------- GameEngine ----------------

def test_engine_init_defaults():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")
    assert engine.white_check is False
    assert engine.black_check is False


def test_engine_get_turn_alternates_with_players():
    engine = GameEngine(Board(), "computer", "human")
    assert engine.get_turn() == ("white", "computer")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    moves = engine.get_moves((0, 1))
    assert {m[1] for m in moves} == {(2, 0), (2, 2)}


def test_num_alph_conversion():
    result = GameEngine.num_alph({1: ((0, 1), (2, 2), "base"),
                                  2: ((1, 7), (3, 7), "base")})
    assert result == {1: ["B1", "C3"], 2: ["H2", "H4"]}


def test_choose_move_computer_mode_updates_state():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((0, 1), (2, 2), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[2, 2], Knight)
    assert engine.get_turn()[0] == "black"
    assert engine.white_check is False
    assert engine.black_check is False


def test_choose_move_errors():
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    engine = GameEngine(Board(), "human", "human")
    with pytest.raises(ValueError):
        engine.choose_move(((0, 1), (2, 2), "base"))
    engine = GameEngine(Board(), "alien", "human")
    with pytest.raises(ValueError):
        engine.choose_move(((0, 1), (2, 2), "base"))


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.Board.update_board(((0, 1), (2, 2), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)
    assert isinstance(engine.Board.board[0, 1], Knight)


def test_is_check_detects_check():
    b = clean_board()
    b[0, 4] = King((0, 4), "white")
    b[7, 7] = King((7, 7), "black")
    b[3, 4] = Rook((3, 4), "black")
    engine = GameEngine(Board(starting_board=b))
    engine.is_check()
    assert engine.white_check is True
    assert engine.black_check is False


def test_is_check_false_at_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.white_check is False
    assert engine.black_check is False


def test_stalemate_3states_triggers_exit_on_repeated_states():
    engine = GameEngine(Board())
    engine.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_stalemate_3states_no_exit_with_few_states():
    engine = GameEngine(Board())
    engine.is_stalemate_3states("white")  # should not raise
    assert engine.compare_states(Board(), Board()) is True
