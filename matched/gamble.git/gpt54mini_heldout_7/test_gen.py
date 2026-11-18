import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand


def test_suits_all_sorted_and_dict():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]

    suit_dict = Card.Suits.dict()
    assert set(suit_dict.keys()) == {"S", "C", "D", "H"}
    assert suit_dict["S"] is Card.Suits.SPADES
    assert suit_dict["H"] is Card.Suits.HEARTS


def test_values_all_sorted_and_dict():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert [v.char for v in values] == ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

    value_dict = Card.Values.dict()
    assert set(value_dict.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert value_dict["A"] is Card.Values.ACE
    assert value_dict["K"] is Card.Values.KING


@pytest.mark.parametrize(
    "text, expected_value, expected_suit",
    [
        ("AS", Card.Values.ACE, Card.Suits.SPADES),
        ("td", Card.Values.TEN, Card.Suits.DIAMONDS),
        ("qH", Card.Values.QUEEN, Card.Suits.HEARTS),
    ],
)
def test_card_get_parses_valid_cards_case_insensitive(text, expected_value, expected_suit):
    card = Card.get(text)
    assert card.value is expected_value
    assert card.suit is expected_suit
    assert str(card) == f"{expected_value.char}{expected_suit.symbol}"


@pytest.mark.parametrize(
    "text, message",
    [
        ("A", "Too many characters for a card!"),
        ("ASS", "Too many characters for a card!"),
        ("1S", "Invalid value for card!"),
        ("AX", "Invalid suit for card!"),
    ],
)
def test_card_get_invalid_inputs_raise_invalidcard(text, message):
    with pytest.raises(InvalidCard, match=message):
        Card.get(text)


def test_card_properties_and_string_representations():
    card = Card.get("AH")
    assert card.full_name == "ace of hearts"
    assert card.color == Card.RED
    assert card.is_red is True
    assert card.is_black is False
    assert str(card) == "A♥"
    assert repr(card) == "<Card:A♥>"


def test_card_unicode_for_face_cards_and_number_cards():
    assert Card.get("AS").unicode == "🂡"
    assert Card.get("TS").unicode == "🂺"
    assert Card.get("QS").unicode == "🂭"
    assert Card.get("KS").unicode == "🂮"


def test_card_comparisons_and_equality():
    low = Card.get("2S")
    high = Card.get("KH")
    same = Card.get("2S")
    different_suit = Card.get("2H")

    assert low < high
    assert high > low
    assert low <= same
    assert high >= same
    assert low == same
    assert low != high
    assert low != different_suit
    assert low != "2S"


def test_deck_default_initialization_has_52_cards_and_known_order():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.bottom == Card.get("KS")
    assert deck.top == Card.get("AS")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_only_cards():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck
    assert object() not in deck


def test_deck_draw_one_updates_state_and_returns_top_card():
    deck = Deck(shuffle=False)
    top_before = deck.top
    drawn = deck.draw()

    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck


def test_deck_draw_multiple_returns_list_in_reverse_top_order():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)

    assert isinstance(drawn, list)
    assert drawn == [Card.get("AS"), Card.get("KS"), Card.get("QS")]
    assert deck.draws == 3
    assert deck.cards_left == 49
    assert deck.top == Card.get("JS")


def test_deck_draw_hand_returns_hand_and_consumes_cards():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.draws == 5
    assert deck.cards_left == 47
    assert hand.cards == [
        Card.get("AS"),
        Card.get("KS"),
        Card.get("QS"),
        Card.get("JS"),
        Card.get("TS"),
    ]


def test_deck_shuffle_calls_random_shuffle_expected_number_of_times(monkeypatch):
    deck = Deck(shuffle=False)

    calls = []

    def fake_shuffle(cards):
        calls.append(list(cards))
        cards.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    original_top = deck.top
    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert len(calls) == 3
    assert deck.top == original_top  # reversed 3 times -> same top as original


def test_custom_deck_initialization_uses_given_cards_without_modifying_order():
    cards = [Card.get("2S"), Card.get("AH"), Card.get("TD")]
    deck = Deck(cards=list(cards), shuffle=False)

    assert deck.cards == cards
    assert deck.bottom == Card.get("2S")
    assert deck.top == Card.get("TD")
    assert deck.cards_left == 3


def test_empty_custom_deck_draw_raises_indexerror():
    deck = Deck(cards=[], shuffle=False)

    with pytest.raises(IndexError):
        deck.draw()
