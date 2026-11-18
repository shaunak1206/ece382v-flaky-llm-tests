import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.king import King
from darkchess.src.rook import Rook
from darkchess.src.queen import Queen


def _empty_board():
    return clean_board()


def test_clean_board_returns_8x8_dummy_board():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert board[i, j].team is None
            assert board[i, j].symbol == "   "
            assert board[i, j].get_position() == (i, j)


def test_board_default_starting_position_has_expected_pieces():
    board = Board()
    assert board.turn == "white"
    assert board.board[0, 0].symbol == "wR "
    assert board.board[0, 1].symbol == "wk "
    assert board.board[0, 3].symbol == "wQ "
    assert board.board[0, 4].symbol == "wK "
    assert board.board[1, 0].symbol == "wp "
    assert board.board[6, 0].symbol == "bp "
    assert board.board[7, 7].symbol == "bR "


def test_board_copy_is_deep_and_equal_matches_same_layout():
    board = Board()
    copied = board.copy()

    assert board.equal(copied)
    assert copied.equal(board)
    assert board is not copied
    assert board.board[0, 1] is not copied.board[0, 1]
    assert board.board[0, 1].symbol == copied.board[0, 1].symbol
    assert board.board[0, 1].get_position() == copied.board[0, 1].get_position()


def test_board_update_base_moves_piece_and_switches_turn():
    board = Board(_empty_board())
    knight = Knight((0, 1), "white")
    board.board[0, 1] = knight

    board.update_board(((0, 1), (2, 2), "base"), "white")

    assert board.turn == "black"
    assert board.board[0, 1].symbol == "   "
    assert board.board[2, 2].symbol == "wk "
    assert board.board[2, 2].moved is True
    assert board.board[2, 2].get_position() == (2, 2)


def test_board_update_en_passant_clears_captured_square():
    board = Board(_empty_board())
    board.board[4, 4] = Knight((4, 4), "white")
    board.board[4, 5] = Knight((4, 5), "black")

    board.update_board(((4, 4), (5, 5), "enpasse"), "white")

    assert board.turn == "black"
    assert board.board[4, 4].symbol == "   "
    assert board.board[5, 5].symbol == "wk "
    assert board.board[4, 5].symbol == "   "


def test_board_update_castling_kingside_moves_rook_correctly():
    board = Board(_empty_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[0, 7] = Rook((0, 7), "white")

    board.update_board(((0, 4), (0, 6), "castling"), "white")

    assert board.turn == "black"
    assert board.board[0, 4].symbol == "   "
    assert board.board[0, 6].symbol == "wK "
    assert board.board[0, 5].symbol == "wR "
    assert board.board[0, 7].symbol == "   "
    assert board.board[0, 6].moved is True
    assert board.board[0, 5].moved is True


def test_board_update_castling_queenside_moves_rook_correctly():
    board = Board(_empty_board())
    board.board[7, 4] = King((7, 4), "black")
    board.board[7, 0] = Rook((7, 0), "black")

    board.update_board(((7, 4), (7, 2), "castling"), "black")

    assert board.turn == "white"
    assert board.board[7, 4].symbol == "   "
    assert board.board[7, 2].symbol == "bK "
    assert board.board[7, 3].symbol == "bR "
    assert board.board[7, 0].symbol == "   "


def test_board_update_evolve_promotes_to_queen():
    board = Board(_empty_board())
    board.board[6, 0] = Knight((6, 0), "white")

    board.update_board(((6, 0), (7, 0), "evolve"), "white")

    assert board.turn == "black"
    assert board.board[6, 0].symbol == "   "
    assert board.board[7, 0].symbol == "wQ "


def test_board_str_contains_coordinates_and_piece_symbols():
    board = Board()
    board_str = str(board)
    assert "A  B  C  D  E  F  G  H" in board_str
    assert "8 " in board_str and "1 " in board_str
    assert "wK " in board_str
    assert "bK " in board_str


def test_knight_get_moves_white_from_center_all_eight_moves():
    board = Board(_empty_board())
    knight = Knight((3, 3), "white")
    board.board[3, 3] = knight

    moves = knight.get_moves([board])

    expected_ends = {
        (1, 2), (1, 4), (2, 1), (4, 1),
        (5, 2), (5, 4), (4, 5), (2, 5),
    }
    assert len(moves) == 8
    assert {move[1] for move in moves} == expected_ends
    assert all(move[2] == "base" for move in moves)


def test_knight_get_moves_black_blocks_same_team_and_allows_capture():
    board = Board(_empty_board())
    knight = Knight((3, 3), "black")
    board.board[3, 3] = knight
    board.board[1, 2] = Dummy((1, 2), "black")  # blocked
    board.board[1, 4] = Dummy((1, 4), "white")  # capturable

    moves = knight.get_moves([board])
    ends = {move[1] for move in moves}

    assert (1, 2) not in ends
    assert (1, 4) in ends
    assert len(moves) == 7


def test_knight_get_moves_on_edge_has_only_legal_moves():
    board = Board(_empty_board())
    knight = Knight((0, 0), "white")
    board.board[0, 0] = knight

    moves = knight.get_moves([board])

    assert {move[1] for move in moves} == {(1, 2), (2, 1)}
    assert len(moves) == 2


def test_knight_get_symbol_invalid_team_raises():
    knight = Knight((0, 0), "green")
    with pytest.raises(ValueError):
        knight.get_symbol()


def test_gameengine_num_alph_converts_coordinates():
    move_dict = {
        1: ((0, 0), (2, 1), "base"),
        2: ((7, 7), (5, 6), "base"),
    }
    converted = GameEngine.num_alph(move_dict)

    assert converted == {
        1: ["A1", "B3"],
        2: ["H8", "G6"],
    }


def test_gameengine_get_turn_and_get_moves_use_current_board_state():
    board = Board(_empty_board())
    board.board[3, 3] = Knight((3, 3), "white")
    engine = GameEngine(board, player1="human", player2="computer")

    team, player = engine.get_turn()
    assert team == "white"
    assert player == "human"

    moves = engine.get_moves((3, 3))
    assert {move[1] for move in moves} == {
        (1, 2), (1, 4), (2, 1), (4, 1),
        (5, 2), (5, 4), (4, 5), (2, 5),
    }


def test_gameengine_is_check_detects_knight_attack():
    board = Board(_empty_board())
    board.board[0, 4] = King((0, 4), "white")
    board.board[2, 5] = Knight((2, 5), "black")

    engine = GameEngine(board)
    engine.is_check()

    assert engine.white_check is True
    assert engine.black_check is False


def test_gameengine_undo_move_restores_previous_state():
    board = Board(_empty_board())
    board.board[0, 0] = Knight((0, 0), "white")
    engine = GameEngine(board)

    engine.Board.update_board(((0, 0), (1, 2), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    engine.undo_move()

    assert engine.Board.board[0, 0].symbol == "wk "
    assert engine.Board.board[1, 2].symbol == "   "
    assert engine.game_states[-1].board[0, 0].symbol == "wk "


def test_gameengine_computer_mode_requires_move_and_human_mode_rejects_move(monkeypatch):
    board = Board(_empty_board())
    engine = GameEngine(board, player1="human", player2="computer")

    with pytest.raises(ValueError):
        engine.choose_move(move=((0, 0), (1, 2), "base"))

    engine.move_counter = 1  # black to play, computer mode
    with pytest.raises(ValueError):
        engine.choose_move(move=None)


def test_gameengine_is_stalemate_3states_no_exit_when_pattern_not_met(monkeypatch):
    board = Board()
    engine = GameEngine(board)

    called = []

    def fake_exit(code):
        called.append(code)
        raise AssertionError("sys.exit should not be called")

    monkeypatch.setattr("darkchess.src.gamengine.sys.exit", fake_exit)

    # Short history: should do nothing.
    engine.is_stalemate_3states("white")
    assert called == []


def test_board_update_invalid_turn_raises_value_error():
    board = Board(_empty_board())
    board.turn = "neither"
    board.board[0, 1] = Knight((0, 1), "white")

    with pytest.raises(ValueError):
        board.update_board(((0, 1), (2, 2), "base"), "white")
