import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.queen import Queen
from darkchess.src.pawn import Pawn


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].get_position() == (i, j)
            assert board[i, j].team is None


def test_board_default_start_has_expected_pieces_and_turn():
    board = Board()
    assert board.turn == "white"
    assert board.board[0, 0].symbol == "wr "
    assert board.board[0, 1].symbol == "wk "
    assert board.board[0, 3].symbol == "wq "
    assert board.board[0, 4].symbol == "wK "
    assert board.board[7, 4].symbol == "bK "
    assert board.board[6, 0].symbol == "bp "
    assert board.board[1, 7].symbol == "bp "


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    s = str(board)
    assert "A  B  C  D  E  F  G  H" in s
    assert "8 " in s and "1 " in s
    assert "wk " in s
    assert "bK " in s


def test_board_update_base_moves_piece_and_toggles_turn():
    board = Board(clean_board())
    board.board[0, 1] = Knight((0, 1), "white")
    move = ((0, 1), (2, 2), "base")

    board.update_board(move, "white")

    assert isinstance(board.board[2, 2], Knight)
    assert board.board[2, 2].team == "white"
    assert board.board[2, 2].get_position() == (2, 2)
    assert isinstance(board.board[0, 1], Dummy)
    assert board.turn == "black"


def test_board_update_enpassant_clears_captured_square():
    board = Board(clean_board())
    board.board[4, 4] = Pawn((4, 4), "white")
    board.board[4, 5] = Pawn((4, 5), "black")

    board.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert board.board[5, 5].team == "white"
    assert board.board[5, 5].get_position() == (5, 5)
    assert isinstance(board.board[4, 4], Dummy)
    assert isinstance(board.board[4, 5], Dummy)
    assert board.turn == "black"


def test_board_update_castling_moves_king_and_rook_queen_side():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 0] = Rook((0, 0), "white")

    board.update_board(((0, 4), (0, 2), "castling"), "white")

    assert board.board[0, 2].symbol == "wK "
    assert board.board[0, 2].get_position() == (0, 2)
    assert board.board[0, 3].symbol == "wr "
    assert board.board[0, 3].get_position() == (0, 3)
    assert isinstance(board.board[0, 4], Dummy)
    assert isinstance(board.board[0, 0], Dummy)


def test_board_update_evolve_replaces_pawn_with_queen():
    board = Board(clean_board())
    board.board[6, 0] = Pawn((6, 0), "white")

    board.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert isinstance(board.board[7, 0], Queen)
    assert board.board[7, 0].team == "white"
    assert board.board[7, 0].get_position() == (7, 0)
    assert isinstance(board.board[6, 0], Dummy)


def test_board_copy_is_deep_and_equal_matches_same_positions():
    board = Board()
    copied = board.copy()

    assert board is not copied
    assert board.equal(copied)

    copied.board[0, 1] = Dummy((0, 1))
    assert not board.equal(copied)
    assert board.board[0, 1].symbol == "wk "


def test_knight_get_symbol_for_both_teams():
    assert Knight((0, 0), "white").get_symbol() == "wk "
    assert Knight((0, 0), "black").get_symbol() == "bk "


def test_knight_get_moves_center_has_eight_moves_on_empty_board():
    board = Board(clean_board())
    knight = Knight((4, 4), "white")
    board.board[4, 4] = knight

    moves = knight.get_moves([board])

    assert len(moves) == 8
    end_positions = {m[1] for m in moves}
    assert end_positions == {
        (2, 3), (2, 5), (3, 2), (3, 6),
        (5, 2), (5, 6), (6, 3), (6, 5),
    }


def test_knight_does_not_capture_own_piece_and_blocks_square():
    board = Board(clean_board())
    knight = Knight((4, 4), "white")
    board.board[4, 4] = knight
    board.board[6, 5] = Pawn((6, 5), "white")
    board.board[2, 3] = Pawn((2, 3), "black")

    moves = knight.get_moves([board])
    end_positions = {m[1] for m in moves}

    assert (6, 5) not in end_positions
    assert (2, 3) in end_positions


def test_gameengine_get_turn_and_num_alph():
    engine = GameEngine(Board(clean_board()), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")

    move_dict = {1: ((0, 1), (2, 2), "base"), 2: ((7, 6), (5, 5), "base")}
    assert engine.num_alph(move_dict) == {
        1: ["B1", "C3"],
        2: ["G8", "F6"],
    }


def test_gameengine_compare_states_and_is_check_detects_check():
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    board.board[7, 0] = Rook((7, 0), "black")
    engine = GameEngine(board)

    assert engine.compare_states(engine.game_states[0], engine.game_states[0])

    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_gameengine_undo_move_restores_previous_state():
    board = Board()
    engine = GameEngine(board)

    original = engine.Board.copy()
    move = ((1, 0), (2, 0), "base")
    engine.Board.update_board(move, "white")
    engine.game_states.append(engine.Board.copy())

    engine.undo_move()

    assert engine.Board.equal(original)
    assert len(engine.game_states) == 1


def test_gameengine_stalemate_detection_calls_exit_when_no_moves(monkeypatch):
    board = Board(clean_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[7, 4] = King((7, 4), "black")
    engine = GameEngine(board)

    monkeypatch.setattr(engine, "all_moves", lambda: [])

    called = {"exit": False}

    def fake_exit(code=0):
        called["exit"] = True
        raise SystemExit(code)

    monkeypatch.setattr("sys.exit", fake_exit)

    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")
    assert called["exit"] is True
