"""Unit tests for gamble.models.cards: Card and Deck."""
import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"


def test_get_parses_value_and_suit_case_insensitively():
    card = Card.get("TH")
    assert card.value is Card.Values.TEN
    assert card.suit is Card.Suits.HEARTS
    lower = Card.get("qd")
    assert lower.value is Card.Values.QUEEN
    assert lower.suit is Card.Suits.DIAMONDS


def test_get_rejects_wrong_length():
    for text in ["", "A", "AAA", "10H"]:
        with pytest.raises(InvalidCard):
            Card.get(text)


def test_get_rejects_invalid_value_or_suit():
    with pytest.raises(InvalidCard):
        Card.get("1H")
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_equality_requires_same_value_and_suit():
    assert Card.get("KD") == Card.get("kd")
    assert Card.get("KD") != Card.get("KH")
    assert Card.get("KD") != Card.get("QD")


def test_equality_with_non_card_is_false():
    assert Card.get("AS") != "AS"
    assert (Card.get("AS") == 0) is False


def test_ordering_operators_compare_by_value():
    assert Card.get("2C") < Card.get("3D")
    assert Card.get("KH") > Card.get("QS")
    assert Card.get("4S") <= Card.get("9S")
    assert Card.get("9S") >= Card.get("4S")
    assert Card.get("JH") <= Card.get("JH")
    assert Card.get("JH") >= Card.get("JH")
    assert not Card.get("5C") < Card.get("5H")
    assert not Card.get("5C") > Card.get("5H")


def test_sorting_orders_cards_by_value():
    cards = [Card.get("KS"), Card.get("2D"), Card.get("AH"), Card.get("TC")]
    assert [c.value.char for c in sorted(cards)] == ["A", "2", "T", "K"]


def test_color_and_is_black_is_red():
    assert Card.get("AS").color == Card.BLACK
    assert Card.get("AC").color == Card.BLACK
    assert Card.get("AD").color == Card.RED
    assert Card.get("AH").color == Card.RED
    assert Card.get("KC").is_black and not Card.get("KC").is_red
    assert Card.get("KD").is_red and not Card.get("KD").is_black


def test_full_name():
    assert Card.get("QD").full_name == "queen of diamonds"
    assert Card.get("2S").full_name == "two of spades"


def test_str_and_repr():
    card = Card.get("TH")
    assert str(card) == "T♥"
    assert repr(card) == "<Card:T♥>"


def test_unicode_skips_knight_for_queen_and_king():
    assert Card.get("AS").unicode == chr(0x1F0A1)
    assert Card.get("JS").unicode == chr(0x1F0AB)
    assert Card.get("QS").unicode == chr(0x1F0AD)
    assert Card.get("KS").unicode == chr(0x1F0AE)


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len({(c.value.name, c.suit.name) for c in deck.cards}) == 52


def test_unshuffled_deck_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_constructor_shuffle_flag():
    assert Deck().shuffles == 1
    assert Deck(shuffle=False).shuffles == 0
    assert Deck().draws == 0
    assert Deck().cards_left == 52


def test_shuffle_keeps_cards_and_changes_order():
    random.seed(1234)
    deck = Deck(shuffle=False)
    original = list(deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert deck.cards != original
    assert sorted(str(c) for c in deck.cards) == sorted(str(c) for c in original)


def test_draw_one_returns_top_card_and_updates_counters():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert Card.get("AS") not in deck


def test_draw_many_returns_list_in_draw_order():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert [str(c) for c in cards] == ["A♠", "2♠", "3♠"]
    assert deck.draws == 3
    assert deck.cards_left == 49


def test_draw_more_than_remaining_raises_index_error():
    deck = Deck(cards=[Card.get("AS"), Card.get("KH")], shuffle=False)
    with pytest.raises(IndexError):
        deck.draw(times=3)


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_draw_hand_of_one_card_wraps_single_draw():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert len(hand) == 1
    assert hand.cards[0] == Card.get("AS")


def test_contains_checks_cards_only():
    deck = Deck(shuffle=False)
    assert Card.get("QH") in deck
    assert "QH" not in deck


def test_deck_str_and_repr_report_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    deck.draw(times=2)
    assert repr(deck) == "<Deck[50]>"


def test_euchre_deck_has_24_cards_from_nine_up():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    assert len({(c.value.name, c.suit.name) for c in deck.cards}) == 24
    assert {c.value.value for c in deck.cards} == {1, 9, 10, 11, 12, 13}
