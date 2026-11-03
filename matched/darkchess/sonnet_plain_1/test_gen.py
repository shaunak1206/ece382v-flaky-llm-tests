import pytest

from darkchess.src.board import Board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.gamengine import GameEngine
from darkchess.src.king import King
from darkchess.src.knight import Knight
from darkchess.src.pawn import Pawn
from darkchess.src.piece import Piece
from darkchess.src.rook import Rook
from darkchess.src.utils import clean_board


def empty_board():
    return Board(starting_board=clean_board())


def end_squares(moves):
    return set(m[1] for m in moves)


# ---------------------------------------------------------------- clean_board

def test_clean_board_is_8x8_of_dummies():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            piece = board[i, j]
            assert isinstance(piece, Dummy)
            assert piece.team is None
            assert piece.get_position() == (i, j)
            assert piece.symbol == "   "


# --------------------------------------------------------------------- Knight

def test_knight_attributes_and_symbols():
    white = Knight((0, 1), "white")
    black = Knight((7, 1), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.symbol == "wk "
    assert black.symbol == "bk "
    assert white.get_position() == (0, 1)
    assert white.moved is False


def test_knight_invalid_team_raises():
    with pytest.raises(ValueError):
        Knight((0, 1), "green")


def test_knight_moves_center_and_corner():
    board = empty_board()
    board.board[3, 3] = Knight((3, 3), "white")
    board.board[0, 0] = Knight((0, 0), "black")
    center = board.board[3, 3].get_moves([board])
    assert len(center) == 8
    assert end_squares(center) == {(1, 2), (1, 4), (2, 1), (4, 1),
                                   (5, 2), (5, 4), (2, 5), (4, 5)}
    assert all(m[0] == (3, 3) and m[2] == "base" for m in center)
    corner = board.board[0, 0].get_moves([board])
    assert end_squares(corner) == {(2, 1), (1, 2)}


def test_knight_blocked_by_own_and_captures_enemy():
    board = empty_board()
    board.board[3, 3] = Knight((3, 3), "white")
    board.board[5, 4] = Pawn((5, 4), "white")
    moves = board.board[3, 3].get_moves([board])
    assert len(moves) == 7
    assert (5, 4) not in end_squares(moves)

    board.board[5, 4] = Pawn((5, 4), "black")
    moves = board.board[3, 3].get_moves([board])
    assert len(moves) == 8
    assert (5, 4) in end_squares(moves)


def test_knight_start_position_moves_and_copy():
    board = Board()
    knight = board.board[0, 1]
    assert end_squares(knight.get_moves([board])) == {(2, 0), (2, 2)}

    knight.moved = True
    clone = knight.copy()
    assert clone is not knight
    assert isinstance(clone, Knight)
    assert clone.team == "white"
    assert clone.get_position() == (0, 1)
    assert clone.moved is True


# ---------------------------------------------------------------------- Board

def test_board_default_start_layout():
    board = Board()
    assert board.turn == "white"
    assert board.board.shape == (8, 8)
    back_rank = ["r", "k", "b", "q", "K", "b", "k", "r"]
    for j in range(8):
        assert board.board[0, j].symbol == "w" + back_rank[j] + " "
        assert board.board[7, j].symbol == "b" + back_rank[j] + " "
        assert board.board[1, j].symbol == "wp "
        assert board.board[6, j].symbol == "bp "
        for i in range(2, 6):
            assert board.board[i, j].team is None
    assert board.board[0, 1].get_name() == "knight"


def test_board_custom_starting_board():
    custom = clean_board()
    custom[3, 3] = Knight((3, 3), "black")
    board = Board(starting_board=custom)
    assert board.board is custom
    assert board.board[3, 3].symbol == "bk "
    assert board.board[0, 0].team is None
    assert board.turn == "white"


def test_update_board_base_and_none_keyword():
    board = Board()
    board.update_board(((1, 4), (3, 4), "base"), "white")
    assert board.board[3, 4].symbol == "wp "
    assert board.board[3, 4].get_position() == (3, 4)
    assert board.board[3, 4].moved is True
    assert isinstance(board.board[1, 4], Dummy)
    assert board.turn == "black"

    board.update_board(((6, 0), (4, 0), None), "black")
    assert board.board[4, 0].symbol == "bp "
    assert board.board[6, 0].symbol == "   "
    assert board.turn == "white"


def test_update_board_enpasse_removes_captured_pawn():
    board = empty_board()
    board.board[4, 3] = Pawn((4, 3), "white")
    board.board[4, 4] = Pawn((4, 4), "black")
    board.update_board(((4, 3), (5, 4), "enpasse"), "white")
    assert board.board[5, 4].symbol == "wp "
    assert board.board[5, 4].get_position() == (5, 4)
    assert board.board[4, 3].symbol == "   "
    assert board.board[4, 4].symbol == "   "
    assert board.turn == "black"


def test_update_board_castling_both_sides():
    board = empty_board()
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")
    board.board[0, 0] = Rook((0, 0), "white")
    board.update_board(((0, 4), (0, 6), "castling"), "white")
    assert board.board[0, 6].symbol == "wK "
    assert board.board[0, 5].symbol == "wr "
    assert board.board[0, 5].get_position() == (0, 5)
    assert board.board[0, 5].moved is True
    assert board.board[0, 7].symbol == "   "
    assert board.board[0, 4].symbol == "   "

    board2 = empty_board()
    board2.board[0, 4] = King((0, 4), "white")
    board2.board[0, 0] = Rook((0, 0), "white")
    board2.update_board(((0, 4), (0, 2), "castling"), "white")
    assert board2.board[0, 2].symbol == "wK "
    assert board2.board[0, 3].symbol == "wr "
    assert board2.board[0, 3].get_position() == (0, 3)
    assert board2.board[0, 0].symbol == "   "


def test_update_board_evolve_promotes_to_queen():
    board = empty_board()
    board.board[6, 0] = Pawn((6, 0), "white")
    board.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert board.board[7, 0].symbol == "wq "
    assert board.board[7, 0].get_name() == "queen"
    assert board.board[6, 0].symbol == "   "
    assert board.turn == "black"


def test_update_board_invalid_turn_raises():
    board = Board()
    board.turn = "green"
    with pytest.raises(ValueError):
        board.update_board(((1, 0), (2, 0), "base"), "white")


def test_board_copy_is_independent_and_equal():
    board = Board()
    clone = board.copy()
    assert clone is not board
    assert clone.turn == board.turn
    assert board.equal(clone)
    assert clone.board[1, 0] is not board.board[1, 0]

    clone.update_board(((1, 0), (3, 0), "base"), "white")
    assert not board.equal(clone)
    assert board.board[1, 0].symbol == "wp "
    assert board.turn == "white"
    assert clone.turn == "black"


def test_board_str_and_view_board(capsys):
    board = Board()
    text = str(board)
    header = "  A  B  C  D  E  F  G  H \n"
    assert text.startswith(header)
    assert text.endswith(header)
    assert len(text.split("\n")) == 11
    assert "wK " in text
    assert "bK " in text
    board.view_board()
    out = capsys.readouterr().out
    assert "wp " in out
    assert "bp " in out


# ----------------------------------------------------------------- GameEngine

def test_engine_init_and_get_turn():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.white_check is False
    assert engine.black_check is False
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "human")

    comp = GameEngine(Board(), "computer", "human")
    assert comp.get_turn() == ("white", "computer")


def test_engine_get_moves_and_all_moves_on_start_board():
    engine = GameEngine(Board())
    assert end_squares(engine.get_moves((1, 0))) == {(2, 0), (3, 0)}
    assert end_squares(engine.get_moves((0, 1))) == {(2, 0), (2, 2)}
    assert engine.get_moves((0, 0)) == []
    assert len(engine.all_moves()) == 20


def test_engine_num_alph_converts_coordinates():
    result = GameEngine.num_alph({1: ((1, 0), (2, 0), "base"),
                                  2: ((0, 6), (2, 7), "base")})
    assert result == {1: ["A2", "A3"], 2: ["G1", "H3"]}


def test_choose_move_error_cases(capsys):
    with pytest.raises(ValueError):
        GameEngine(Board(), "robot", "human").choose_move()
    with pytest.raises(ValueError):
        GameEngine(Board(), "computer", "computer").choose_move()
    with pytest.raises(ValueError):
        GameEngine(Board()).choose_move(((1, 0), (2, 0), "base"))


def test_choose_move_computer_plays_legal_move(capsys):
    engine = GameEngine(Board(), "computer", "computer")
    engine.choose_move(((1, 4), (3, 4), "base"))
    assert engine.move_counter == 1
    assert len(engine.game_states) == 2
    assert engine.Board.board[3, 4].symbol == "wp "
    assert engine.get_turn() == ("black", "computer")
    assert engine.white_check is False
    assert engine.black_check is False
    assert "white to play" in capsys.readouterr().out


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board(), "computer", "computer")
    initial = engine.game_states[0].copy()
    engine.choose_move(((1, 4), (3, 4), "base"))
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert engine.Board.equal(initial)
    assert engine.Board.board[1, 4].symbol == "wp "
    assert engine.Board.board[3, 4].symbol == "   "


def test_is_check_detects_rook_attack():
    board = empty_board()
    board.board[0, 0] = King((0, 0), "white")
    board.board[7, 4] = King((7, 4), "black")
    board.board[3, 4] = Rook((3, 4), "white")
    engine = GameEngine(board)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False

    moves, king_pos = engine._all_moves_minus_king(engine.game_states, "white")
    assert king_pos == (0, 0)
    assert (7, 4) in moves

    # move the rook off the king's file: no more check
    board2 = empty_board()
    board2.board[0, 0] = King((0, 0), "white")
    board2.board[7, 4] = King((7, 4), "black")
    board2.board[3, 3] = Rook((3, 3), "white")
    engine2 = GameEngine(board2)
    engine2.is_check()
    assert engine2.black_check is False
    assert engine2.white_check is False


def test_stalemate_by_repetition_and_compare_states():
    engine = GameEngine(Board())
    engine.is_stalemate_3states("white")  # only one state, must not exit

    other = Board()
    other.update_board(((1, 0), (3, 0), "base"), "white")
    assert engine.compare_states(engine.game_states[0], Board()) is True
    assert engine.compare_states(engine.game_states[0], other) is False

    engine.game_states = [Board() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")
