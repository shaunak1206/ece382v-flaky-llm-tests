import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_returns_sorted_suits():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_suits_dict_maps_char_to_suit():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["S"] is Card.Suits.SPADES
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_returns_sorted_values():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert [v.char for v in values] == ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def test_values_dict_maps_char_to_value():
    values = Card.Values.dict()
    assert set(values.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert values["A"] is Card.Values.ACE
    assert values["K"] is Card.Values.KING


def test_card_get_parses_case_insensitive_text():
    card = Card.get("qh")
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "Q♥"
    assert repr(card) == "<Card:Q♥>"


def test_card_get_rejects_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("10H")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("XH")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("ASX"[:2])  # "AX"


def test_card_color_and_boolean_properties():
    spade = Card.get("AS")
    heart = Card.get("AH")
    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False
    assert heart.color == Card.RED
    assert heart.is_red is True
    assert heart.is_black is False


def test_card_unicode_for_face_cards_and_ace():
    assert Card.get("AS").unicode == "🂡"
    assert Card.get("QS").unicode == "🂭"
    assert Card.get("KS").unicode == "🂮"
    assert Card.get("TH").unicode == "🂺"


def test_card_comparison_and_equality():
    two_clubs = Card.get("2C")
    ten_spades = Card.get("TS")
    another_two_clubs = Card.get("2C")

    assert two_clubs < ten_spades
    assert ten_spades > two_clubs
    assert two_clubs <= another_two_clubs
    assert two_clubs >= another_two_clubs
    assert two_clubs == another_two_clubs
    assert two_clubs != ten_spades
    assert two_clubs != object()


def test_default_deck_has_52_cards_and_known_top_bottom():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert deck.bottom == Card.get("KS")
    assert deck.top == Card.get("AC")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_card_and_rejects_non_card():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_updates_counts_and_removes_from_top():
    deck = Deck(shuffle=False)
    top_before = deck.top

    drawn = deck.draw()
    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck


def test_deck_draw_multiple_cards_returns_list_and_updates_draws():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)

    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert deck.draws == 3
    assert deck.cards_left == 49
    assert drawn[0] == Card.get("AC")
    assert drawn[1] == Card.get("KC")
    assert drawn[2] == Card.get("QC")


def test_deck_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.draws == 5
    assert deck.cards_left == 47
    assert hand.cards[-1] == Card.get("AC")


def test_deck_shuffle_calls_random_shuffle_and_tracks_count(monkeypatch):
    deck = Deck(shuffle=False)
    calls = []

    def fake_shuffle(seq):
        calls.append(list(seq))
        seq.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    original_top = deck.top
    deck.shuffle(times=2)

    assert deck.shuffles == 2
    assert len(calls) == 2
    assert deck.top == original_top  # reverse twice returns original order


def test_euchre_deck_contains_24_cards_and_expected_values():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert deck.bottom == Card.get("9S")
    assert deck.top == Card.get("AC")

    values = {card.value.value for card in deck.cards}
    assert values == {1, 9, 10, 11, 12, 13}
    assert Card.get("8S") not in deck
    assert Card.get("AS") in deck
