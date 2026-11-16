import pytest

from darkchess.src.board import Board
from darkchess.src.gamengine import GameEngine
from darkchess.src.knight import Knight
from darkchess.src.utils import clean_board
from darkchess.src.dummy_piece import Dummy
from darkchess.src.pawn import Pawn
from darkchess.src.queen import Queen
from darkchess.src.king import King
from darkchess.src.rook import Rook


def empty_board():
    return Board(starting_board=clean_board())


def place(board, pos, piece):
    board.board[pos] = piece
    return board


def ends(moves):
    return {move[1] for move in moves}


def test_clean_board_shape_and_dummies():
    board = clean_board()
    assert board.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            assert isinstance(board[i, j], Dummy)
            assert board[i, j].team is None
            assert board[i, j].get_position() == (i, j)


def test_default_board_piece_placement():
    b = Board()
    assert b.turn == "white"
    assert isinstance(b.board[0, 0], Rook)
    assert isinstance(b.board[0, 1], Knight)
    assert isinstance(b.board[0, 3], Queen)
    assert isinstance(b.board[0, 4], King)
    assert isinstance(b.board[1, 4], Pawn)
    assert b.board[0, 4].team == "white"
    assert isinstance(b.board[7, 4], King)
    assert isinstance(b.board[6, 4], Pawn)
    assert b.board[7, 4].team == "black"
    assert isinstance(b.board[3, 3], Dummy)


def test_update_board_base_move_moves_piece_and_toggles_turn():
    b = Board()
    b.update_board(((0, 1), (2, 2), "base"), "white")
    assert isinstance(b.board[2, 2], Knight)
    assert b.board[2, 2].get_position() == (2, 2)
    assert b.board[2, 2].moved is True
    assert isinstance(b.board[0, 1], Dummy)
    assert b.turn == "black"


def test_update_board_enpasse_removes_captured_pawn():
    b = empty_board()
    place(b, (4, 4), Pawn((4, 4), "white"))
    place(b, (4, 3), Pawn((4, 3), "black"))
    b.update_board(((4, 4), (5, 3), "enpasse"), "white")
    assert isinstance(b.board[5, 3], Pawn)
    assert b.board[5, 3].team == "white"
    assert isinstance(b.board[4, 4], Dummy)
    assert isinstance(b.board[4, 3], Dummy)


def test_update_board_castling_kingside_moves_rook():
    b = empty_board()
    place(b, (0, 4), King((0, 4), "white"))
    place(b, (0, 7), Rook((0, 7), "white"))
    b.update_board(((0, 4), (0, 6), "castling"), "white")
    assert isinstance(b.board[0, 6], King)
    assert isinstance(b.board[0, 5], Rook)
    assert b.board[0, 5].get_position() == (0, 5)
    assert b.board[0, 5].moved is True
    assert isinstance(b.board[0, 7], Dummy)
    assert isinstance(b.board[0, 4], Dummy)


def test_update_board_evolve_creates_queen():
    b = empty_board()
    place(b, (6, 0), Pawn((6, 0), "white"))
    b.update_board(((6, 0), (7, 0), "evolve"), "white")
    assert isinstance(b.board[7, 0], Queen)
    assert b.board[7, 0].team == "white"
    assert isinstance(b.board[6, 0], Dummy)


def test_board_copy_is_independent():
    b = Board()
    b2 = b.copy()
    assert b2.turn == b.turn
    assert b2.equal(b)
    b2.board[0, 0] = Dummy((0, 0), None)
    assert b.board[0, 0].symbol != b2.board[0, 0].symbol
    assert b.board[0, 0].get_position() == (0, 0)


def test_board_equal_detects_difference():
    b = Board()
    b2 = b.copy()
    assert b.equal(b2)
    b2.update_board(((1, 4), (3, 4), "base"), "white")
    assert not b.equal(b2)


def test_knight_moves_from_center_on_empty_board():
    b = empty_board()
    place(b, (4, 4), Knight((4, 4), "white"))
    moves = b.board[4, 4].get_moves([b])
    assert len(moves) == 8
    assert ends(moves) == {(2, 3), (2, 5), (3, 2), (3, 6),
                           (5, 2), (5, 6), (6, 3), (6, 5)}
    for move in moves:
        assert move[0] == (4, 4)
        assert move[2] == "base"


def test_knight_moves_from_corner_on_empty_board():
    b = empty_board()
    place(b, (0, 0), Knight((0, 0), "white"))
    moves = b.board[0, 0].get_moves([b])
    assert ends(moves) == {(2, 1), (1, 2)}


def test_knight_excludes_own_piece_and_includes_enemy():
    b = empty_board()
    place(b, (4, 4), Knight((4, 4), "white"))
    place(b, (2, 3), Pawn((2, 3), "white"))
    place(b, (6, 5), Knight((6, 5), "black"))
    moves = b.board[4, 4].get_moves([b])
    result = ends(moves)
    assert (2, 3) not in result
    assert (6, 5) in result


def test_knight_default_start_moves():
    b = Board()
    assert ends(b.board[0, 1].get_moves([b])) == {(2, 0), (2, 2)}
    assert ends(b.board[7, 1].get_moves([b])) == {(5, 0), (5, 2)}


def test_knight_name_value_and_symbol():
    white = Knight((0, 0), "white")
    black = Knight((7, 7), "black")
    assert white.get_name() == "knight"
    assert white.get_value() == 3
    assert white.get_symbol() == "wk "
    assert black.get_symbol() == "bk "


def test_knight_invalid_team_raises():
    k = Knight((0, 0), "red")
    with pytest.raises(ValueError):
        k.get_symbol()
    b = Board(starting_board=clean_board())
    with pytest.raises(ValueError):
        k.get_moves([b])


def test_is_check_detects_knight_attack():
    b = empty_board()
    place(b, (4, 4), King((4, 4), "black"))
    place(b, (2, 3), Knight((2, 3), "white"))
    engine = GameEngine(b)
    engine.is_check()
    assert engine.black_check is True
    assert engine.white_check is False


def test_game_engine_init_defaults():
    engine = GameEngine(Board())
    assert engine.move_counter == 0
    assert len(engine.game_states) == 1
    assert engine.player1 == "human"
    assert engine.player2 == "human"
    assert engine.black_check is False
    assert engine.white_check is False


def test_get_turn_alternates_between_players():
    engine = GameEngine(Board(), player1="human", player2="computer")
    assert engine.get_turn() == ("white", "human")
    engine.move_counter = 1
    assert engine.get_turn() == ("black", "computer")


def test_num_alph_converts_moves_to_chess_notation():
    move_dict = {1: ((1, 4), (3, 4), "base"), 2: ((0, 1), (2, 2), "base")}
    result = GameEngine.num_alph(move_dict)
    assert result == {1: ["E2", "E4"], 2: ["B1", "C3"]}


def test_undo_move_restores_previous_state():
    engine = GameEngine(Board())
    engine.Board.update_board(((0, 1), (2, 2), "base"), "white")
    engine.game_states.append(engine.Board.copy())
    assert len(engine.game_states) == 2
    engine.undo_move()
    assert len(engine.game_states) == 1
    assert isinstance(engine.Board.board[0, 1], Knight)
    assert engine.Board.board[0, 1].symbol == "wk "


def test_stalemate_3states_repetition_exits():
    engine = GameEngine(Board())
    engine.game_states = [engine.Board.copy() for _ in range(9)]
    with pytest.raises(SystemExit):
        engine.is_stalemate_3states("white")


def test_stalemate_no_move_exits_when_no_pieces():
    engine = GameEngine(Board(starting_board=clean_board()))
    with pytest.raises(SystemExit):
        engine.is_stalemate_no_move("white")


def test_choose_move_computer_plays_given_move():
    engine = GameEngine(Board(), player1="computer", player2="computer")
    engine.choose_move(((0, 1), (2, 2), "base"))
    assert engine.move_counter == 1
    assert isinstance(engine.Board.board[2, 2], Knight)
    assert engine.Board.turn == "black"


def test_choose_move_human_with_move_raises():
    engine = GameEngine(Board())
    with pytest.raises(ValueError):
        engine.choose_move(((0, 1), (2, 2), "base"))


def test_choose_move_human_plays_via_input(monkeypatch):
    responses = iter(["B1", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    engine = GameEngine(Board())
    engine.choose_move()
    assert engine.move_counter == 1
    assert isinstance(engine.Board.board[2, 0], Knight)
    assert engine.Board.board[2, 0].team == "white"
    assert isinstance(engine.Board.board[0, 1], Dummy)
