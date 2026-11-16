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
    return Board(clean_board())


def place(board, piece, pos):
    board.board[pos] = piece
    return piece


def end_squares(moves):
    return set(m[1] for m in moves)


# ---------------- clean_board ----------------

def test_clean_board_shape_and_contents():
    b = clean_board()
    assert b.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(b[i, j], Dummy)
            assert b[i, j].team is None


def test_clean_board_returns_independent_arrays():
    a = clean_board()
    b = clean_board()
    assert a is not b
    a[0, 0] = Knight((0, 0), "white")
    assert isinstance(b[0, 0], Dummy)


# ---------------- Knight ----------------

def test_knight_basic_attributes():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_moves_center_empty_board():
    board = empty_board()
    knight = place(board, Knight((3, 3), "white"), (3, 3))
    moves = knight.get_moves([board])
    assert len(moves) == 8
    assert end_squares(moves) == {(1, 2), (1, 4), (2, 1), (4, 1),
                                  (5, 2), (5, 4), (4, 5), (2, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_moves_corner():
    board = empty_board()
    knight = place(board, Knight((0, 0), "black"), (0, 0))
    moves = knight.get_moves([board])
    assert end_squares(moves) == {(2, 1), (1, 2)}


def test_knight_start_position_moves_white_and_black():
    board = Board()
    white = board.board[0, 1]
    black = board.board[7, 1]
    assert end_squares(white.get_moves([board])) == {(2, 0), (2, 2)}
    assert end_squares(black.get_moves([board])) == {(5, 0), (5, 2)}


def test_knight_can_capture_enemy_but_not_friend():
    board = empty_board()
    knight = place(board, Knight((3, 3), "white"), (3, 3))
    place(board, Pawn((5, 4), "black"), (5, 4))
    place(board, Pawn((5, 2), "white"), (5, 2))
    ends = end_squares(knight.get_moves([board]))
    assert (5, 4) in ends
    assert (5, 2) not in ends
    assert len(ends) == 7


def test_knight_copy_is_independent():
    k = Knight((3, 3), "white")
    k.moved = True
    c = k.copy()
    assert c is not k
    assert isinstance(c, Knight)
    assert c.team == "white"
    assert c.get_position() == (3, 3)
    assert c.moved is True
    c.set_position((4, 4))
    assert k.get_position() == (3, 3)


# ---------------- Board ----------------

def test_board_default_setup():
    board = Board()
    assert board.turn == "white"
    white = black = empty = 0
    for i in range(8):
        for j in range(8):
            team = board.board[i, j].team
            if team == "white":
                white += 1
            elif team == "black":
                black += 1
            else:
                empty += 1
    assert (white, black, empty) == (16, 16, 32)
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[7, 0], Rook)
    for j in range(8):
        assert isinstance(board.board[1, j], Pawn)
        assert isinstance(board.board[6, j], Pawn)


def test_board_custom_starting_board_used():
    arr = clean_board()
    board = Board(arr)
    assert board.board is arr
    assert board.turn == "white"


def test_update_board_base_move():
    board = Board()
    board.update_board(((1, 0), (3, 0), "base"), "white")
    assert isinstance(board.board[3, 0], Pawn)
    assert board.board[3, 0].get_position() == (3, 0)
    assert board.board[3, 0].moved is True
    assert isinstance(board.board[1, 0], Dummy)
    assert board.turn == "black"
    board.update_board(((6, 0), (4, 0), "base"), "black")
    assert board.turn == "white"


def test_update_board_none_keyword_treated_as_base():
    board = Board()
    board.update_board(((0, 1), (2, 2), None), "white")
    assert isinstance(board.board[2, 2], Knight)
    assert isinstance(board.board[0, 1], Dummy)
    assert board.turn == "black"


def test_update_board_enpassant_removes_captured_pawn():
    board = empty_board()
    place(board, Pawn((4, 3), "white"), (4, 3))
    place(board, Pawn((4, 4), "black"), (4, 4))
    board.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert isinstance(board.board[5, 4], Pawn)
    assert board.board[5, 4].team == "white"
    assert isinstance(board.board[4, 3], Dummy)
    assert isinstance(board.board[4, 4], Dummy)


def test_update_board_castling_kingside_and_queenside():
    board = empty_board()
    place(board, King((0, 4), "white"), (0, 4))
    place(board, Rook((0, 7), "white"), (0, 7))
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert board.board[0, 5].get_position() == (0, 5)
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)

    board2 = empty_board()
    place(board2, King((0, 4), "white"), (0, 4))
    place(board2, Rook((0, 0), "white"), (0, 0))
    board2.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(board2.board[0, 2], King)
    assert isinstance(board2.board[0, 3], Rook)
    assert isinstance(board2.board[0, 0], Dummy)


def test_update_board_evolve_creates_queen():
    board = empty_board()
    place(board, Pawn((6, 0), "white"), (6, 0))
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert isinstance(board.board[6, 0], Dummy)
    assert board.turn == "black"


def test_board_copy_and_equal():
    board = Board()
    other = board.copy()
    assert other is not board
    assert other.board is not board.board
    assert board.equal(other)
    assert other.turn == board.turn
    other.update_board(((1, 0), (3, 0), "base"), "white")
    assert not board.equal(other)
    assert isinstance(board.board[1, 0], Pawn)
    assert board.turn == "white"


def test_board_str_layout():
    text = str(Board())
    lines = text.split("\n")
    header = "  A  B  C  D  E  F  G  H "
    assert lines[0] == header
    assert lines[-2] == header
    assert lines[1].startswith("8 ")
    assert lines[1].endswith(" 8")
    assert lines[8].startswith("1 ")
    assert lines[8].endswith(" 1")
    assert len(lines) == 11


# ---------------- GameEngine ----------------

def test_engine_initial_state():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")
    assert engine.black_check is False
    assert engine.white_check is False

    engine2 = GameEngine(Board(), "computer", "human")
    assert engine2.get_turn() == ("white", "computer")
    engine2.move_counter = 1
    assert engine2.get_turn() == ("black", "human")


def test_engine_num_alph_conversion():
    result = GameEngine.num_alph({1: ((1, 0), (3, 0), "base"),
                                  2: ((0, 7), (7, 2), "base")})
    assert result == {1: ["A2", "A4"], 2: ["H1", "C8"]}


def test_engine_get_moves_for_knight():
    engine = GameEngine(Board())
    assert end_squares(engine.get_moves((0, 1))) == {(2, 0), (2, 2)}


def test_engine_computer_move_advances_turn():
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((0, 1), (2, 2), "base"))
    assert engine.move_counter == 1
    assert engine.get_turn()[0] == "black"
    assert len(engine.game_states) == 2
    assert isinstance(engine.Board.board[2, 2], Knight)


def test_engine_choose_move_validation_errors():
    engine = GameEngine(Board(), "computer", "computer")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    engine2 = GameEngine(Board(), "robot", "human")
    with pytest.raises(ValueError):
        engine2.choose_move(((0, 1), (2, 2), "base"))
    engine3 = GameEngine(Board(), "human", "human")
    with pytest.raises(ValueError):
        engine3.choose_move(((0, 1), (2, 2), "base"))


def test_engine_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    original = engine.Board.copy()
    engine.Board.update_board(((1, 0), (3, 0), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(original)
    assert engine.Board.turn == "white"


def test_engine_is_check_detects_knight_attack():
    board = empty_board()
    place(board, King((0, 7), "white"), (0, 7))
    place(board, King((2, 1), "black"), (2, 1))
    place(board, Knight((0, 0), "white"), (0, 0))
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_engine_is_check_false_on_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.black_check is False
    assert engine.white_check is False


def test_engine_stalemate_3states_repetition():
    engine = GameEngine(Board())
    engine.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_engine_stalemate_3states_not_triggered_with_few_states():
    engine = GameEngine(Board())
    engine.game_states = [Board() for _ in range(8)]
    engine.is_stalemate_3states("white")  # must not exit
    assert engine.compare_states(engine.game_states[0], engine.game_states[1])
