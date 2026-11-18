import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand


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
    assert values[0].char == "A"
    assert values[-1].char == "K"


def test_values_dict_maps_char_to_value():
    values = Card.Values.dict()
    assert set(values.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert values["A"] is Card.Values.ACE
    assert values["K"] is Card.Values.KING


def test_card_default_constructor_uses_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_get_parses_text_case_insensitively():
    card = Card.get("qh")
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS
    assert card.full_name == "queen of hearts"
    assert card.color == Card.RED
    assert card.is_red is True
    assert card.is_black is False


def test_card_get_rejects_text_with_wrong_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("10H")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value for card"):
        Card.get("ZH")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit for card"):
        Card.get("AS")


def test_card_unicode_for_face_cards_skips_knight_offset():
    queen_spades = Card.get("QS")
    king_spades = Card.get("KS")
    assert queen_spades.unicode == chr(Card.Suits.SPADES.unicode + 12 + 1)
    assert king_spades.unicode == chr(Card.Suits.SPADES.unicode + 13 + 1)


def test_card_equality_and_ordering():
    low = Card.get("2C")
    same = Card.get("2C")
    high = Card.get("9D")
    other_suit_same_value = Card.get("2D")

    assert low == same
    assert low != high
    assert low != object()
    assert low < high
    assert high > low
    assert low <= same
    assert high >= low
    assert low == other_suit_same_value is False


def test_deck_default_construction_has_52_cards_in_expected_order():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert deck.bottom == Card.get("KS")
    assert deck.top == Card.get("AS")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_contains_only_cards():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_reduces_cards_and_tracks_draws():
    deck = Deck(shuffle=False)
    top_before = deck.top
    drawn = deck.draw()
    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck


def test_deck_draw_multiple_returns_list_and_updates_counter():
    deck = Deck(shuffle=False)
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert deck.draws == 3
    assert deck.cards_left == 49
    assert drawn == [Card.get("AS"), Card.get("KS"), Card.get("QS")]


def test_deck_draw_hand_returns_hand_object():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert hand.cards == [Card.get("AS"), Card.get("KS"), Card.get("QS"), Card.get("JS"), Card.get("TS")]
    assert deck.draws == 5
    assert deck.cards_left == 47


def test_deck_shuffle_calls_random_shuffle_requested_number_of_times(monkeypatch):
    deck = Deck(shuffle=False)
    calls = []

    def fake_shuffle(seq):
        calls.append(list(seq))
        seq.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    original = list(deck.cards)

    deck.shuffle(times=2)

    assert deck.shuffles == 2
    assert len(calls) == 2
    assert deck.cards == original  # reversed twice, ends up same


def test_deck_initialized_with_custom_cards_does_not_generate_full_deck():
    cards = [Card.get("AS"), Card.get("KH")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert Card.get("AS") in deck
    assert Card.get("2S") not in deck
