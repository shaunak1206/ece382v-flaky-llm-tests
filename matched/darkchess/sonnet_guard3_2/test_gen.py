import pytest

from darkchess.src.board import Board
from darkchess.src.bishop import Bishop
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


def ends(moves):
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
    a[0, 0] = Knight((0, 0), "white")
    assert isinstance(b[0, 0], Dummy)
    assert a is not b


# ---------------- Knight ----------------

def test_knight_basic_attributes():
    w = Knight((0, 1), "white")
    b = Knight((7, 1), "black")
    assert w.get_name() == "knight"
    assert w.get_value() == 3
    assert w.get_symbol() == "wk "
    assert b.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    k = Knight((0, 1), "white")
    k.team = "green"
    with pytest.raises(ValueError):
        k.get_symbol()
    with pytest.raises(ValueError):
        k.get_moves([empty_board()])


def test_knight_center_has_eight_moves():
    board = empty_board()
    k = Knight((3, 3), "white")
    board.board[3, 3] = k
    moves = k.get_moves([board])
    assert len(moves) == 8
    assert ends(moves) == {(1, 2), (1, 4), (2, 1), (4, 1),
                           (5, 2), (5, 4), (2, 5), (4, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in moves)


def test_knight_corner_has_two_moves_for_both_teams():
    for team in ("white", "black"):
        board = empty_board()
        k = Knight((0, 0), team)
        board.board[0, 0] = k
        assert ends(k.get_moves([board])) == {(2, 1), (1, 2)}


def test_knight_blocked_by_friend_and_can_capture_enemy():
    board = empty_board()
    k = Knight((0, 0), "white")
    board.board[0, 0] = k
    board.board[2, 1] = Knight((2, 1), "white")
    board.board[1, 2] = Knight((1, 2), "black")
    assert ends(k.get_moves([board])) == {(1, 2)}


def test_knight_copy_is_independent():
    k = Knight((0, 1), "white")
    k.moved = True
    c = k.copy()
    assert isinstance(c, Knight)
    assert c is not k
    assert c.team == "white"
    assert c.moved is True
    assert c.get_position() == (0, 1)


# ---------------- Board ----------------

def test_board_default_start_layout():
    board = Board()
    assert board.turn == "white"
    pieces = [board.board[i, j] for i in range(8) for j in range(8)
              if not isinstance(board.board[i, j], Dummy)]
    assert len(pieces) == 32
    assert sum(1 for p in pieces if p.team == "white") == 16
    assert sum(1 for p in pieces if p.team == "black") == 16
    assert isinstance(board.board[0, 1], Knight)
    assert isinstance(board.board[7, 6], Knight)
    assert isinstance(board.board[0, 3], Queen)
    assert isinstance(board.board[0, 4], King)
    assert isinstance(board.board[0, 0], Rook)
    assert isinstance(board.board[0, 2], Bishop)
    for j in range(8):
        assert isinstance(board.board[1, j], Pawn)
        assert board.board[1, j].team == "white"
        assert isinstance(board.board[6, j], Pawn)
        assert board.board[6, j].team == "black"
    for i in range(2, 6):
        for j in range(8):
            assert isinstance(board.board[i, j], Dummy)


def test_board_update_base_move():
    board = Board()
    board.update_board(((0, 1), (2, 2), "base"), "white")
    assert isinstance(board.board[2, 2], Knight)
    assert board.board[2, 2].get_position() == (2, 2)
    assert board.board[2, 2].moved is True
    assert isinstance(board.board[0, 1], Dummy)
    assert board.turn == "black"
    board.update_board(((7, 1), (5, 2), None), "black")
    assert isinstance(board.board[5, 2], Knight)
    assert board.turn == "white"


def test_board_update_enpasse_removes_captured_pawn():
    board = empty_board()
    board.board[4, 3] = Pawn((4, 3), "white")
    board.board[4, 4] = Pawn((4, 4), "black")
    board.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert isinstance(board.board[5, 4], Pawn)
    assert board.board[5, 4].team == "white"
    assert isinstance(board.board[4, 3], Dummy)
    assert isinstance(board.board[4, 4], Dummy)


def test_board_update_castling_both_sides():
    board = Board()
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(board.board[0, 6], King)
    assert isinstance(board.board[0, 5], Rook)
    assert isinstance(board.board[0, 7], Dummy)
    assert isinstance(board.board[0, 4], Dummy)

    board2 = Board()
    board2.update_board(((0, 4), (0, 2), "castling"), "white")
    assert isinstance(board2.board[0, 2], King)
    assert isinstance(board2.board[0, 3], Rook)
    assert isinstance(board2.board[0, 0], Dummy)
    assert board2.board[0, 3].get_position() == (0, 3)


def test_board_update_evolve_makes_queen():
    board = empty_board()
    board.board[6, 0] = Pawn((6, 0), "white")
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert isinstance(board.board[6, 0], Dummy)


def test_board_copy_and_equal():
    board = Board()
    c = board.copy()
    assert c is not board
    assert c.turn == board.turn
    assert board.equal(c)
    assert c.board[0, 1] is not board.board[0, 1]
    c.update_board(((0, 1), (2, 2), "base"), "white")
    assert not board.equal(c)
    assert isinstance(board.board[0, 1], Knight)
    assert board.turn == "white"


def test_board_str_and_view_board(capsys):
    board = Board()
    s = str(board)
    assert s.startswith("  A  B  C  D  E  F  G  H \n")
    assert s.endswith("  A  B  C  D  E  F  G  H \n")
    assert s.count("\n") == 10
    assert "wk " in s and "bk " in s
    board.view_board()
    out = capsys.readouterr().out
    assert out == s + "\n"


# ---------------- GameEngine ----------------

def test_engine_init_and_turn():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.get_turn() == ("white", "human")
    assert engine.white_check is False
    assert engine.black_check is False
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")
    e2 = GameEngine(Board(), "computer", "human")
    assert e2.get_turn() == ("white", "computer")


def test_engine_num_alph_and_get_moves():
    result = GameEngine.num_alph({1: ((1, 0), (2, 0), "base"),
                                  2: ((0, 7), (7, 3), "base")})
    assert result == {1: ["A2", "A3"], 2: ["H1", "D8"]}

    engine = GameEngine(Board())
    assert ends(engine.get_moves((0, 1))) == {(2, 0), (2, 2)}


def test_engine_computer_move_and_undo(capsys):
    engine = GameEngine(Board(), "computer", "computer")
    initial = engine.Board.copy()
    engine.choose_move(((0, 1), (2, 2), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert engine.get_turn()[0] == "black"
    assert isinstance(engine.Board.board[2, 2], Knight)
    assert not engine.white_check and not engine.black_check

    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(initial)


def test_engine_choose_move_invalid_modes(capsys):
    engine = GameEngine(Board(), "computer", "human")
    with pytest.raises(ValueError):
        engine.choose_move(None)
    engine2 = GameEngine(Board(), "human", "human")
    with pytest.raises(ValueError):
        engine2.choose_move(((0, 1), (2, 2), "base"))
    engine3 = GameEngine(Board(), "alien", "human")
    with pytest.raises(ValueError):
        engine3.choose_move(((0, 1), (2, 2), "base"))


def test_engine_is_check_detects_knight_attack():
    board = empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[2, 2] = King((2, 2), "black")
    board.board[0, 1] = Knight((0, 1), "white")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_engine_is_check_false_on_start():
    engine = GameEngine(Board())
    engine.is_check()
    assert engine.black_check is False
    assert engine.white_check is False


def test_engine_compare_states():
    engine = GameEngine(Board())
    a = Board()
    b = Board()
    assert engine.compare_states(a, b) is True
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert engine.compare_states(a, b) is False


def test_engine_stalemate_3states(capsys):
    engine = GameEngine(Board())
    engine.is_stalemate_3states("white")  # fewer than 9 states: no exit
    for _ in range(8):
        engine.game_states.append(Board())
    assert len(engine.game_states) == 9
    with pytest.raises(SystemExit) as exc:
        engine.is_stalemate_3states("white")
    assert exc.value.code == 0


def test_engine_stalemate_no_move_exits(capsys):
    board = empty_board()
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)
    with pytest.raises(SystemExit) as exc:
        engine.is_stalemate_no_move("white")
    assert exc.value.code == 0
    assert "stalemate" in capsys.readouterr().out


def test_engine_stalemate_no_move_not_triggered_with_moves():
    engine = GameEngine(Board())
    engine.is_stalemate_no_move("white")  # does not raise
    assert len(engine.all_moves()) == 20
