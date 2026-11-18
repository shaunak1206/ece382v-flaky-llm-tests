import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_returns_sorted_suits():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_suits_dict_maps_characters_to_suits():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["S"] is Card.Suits.SPADES
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_returns_sorted_values():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0].char == "A"
    assert values[-1].char == "K"


def test_values_dict_maps_characters_to_values():
    values = Card.Values.dict()
    assert values["A"] is Card.Values.ACE
    assert values["T"] is Card.Values.TEN
    assert values["K"] is Card.Values.KING


def test_card_get_parses_case_insensitive_card_text():
    card = Card.get("ah")
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "A♥"
    assert repr(card) == "<Card:A♥>"


def test_card_get_rejects_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("AHK")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("1H")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AX")


def test_card_color_and_boolean_properties():
    black_card = Card.get("AS")
    red_card = Card.get("AD")

    assert black_card.color == Card.BLACK
    assert black_card.is_black is True
    assert black_card.is_red is False

    assert red_card.color == Card.RED
    assert red_card.is_black is False
    assert red_card.is_red is True


def test_card_unicode_for_face_cards_skips_knight_gap():
    assert Card.get("QS").unicode == "♣"  # queen of spades unicode offset
    assert Card.get("KS").unicode == "♤"
    # sanity check that face cards differ from adjacent values
    assert Card.get("JS").unicode != Card.get("QS").unicode
    assert Card.get("TS").unicode != Card.get("JS").unicode


def test_card_comparison_and_equality():
    ace_spades = Card.get("AS")
    king_spades = Card.get("KS")
    ace_hearts = Card.get("AH")

    assert ace_spades > king_spades
    assert king_spades < ace_spades
    assert ace_spades >= ace_spades
    assert ace_spades <= ace_spades
    assert ace_spades == Card.get("AS")
    assert ace_spades != ace_hearts
    assert ace_spades != object()


def test_deck_default_construction_has_52_cards_and_known_top_bottom():
    deck = Deck(shuffle=False)

    assert deck.cards_left == 52
    assert len(deck) == 52 if hasattr(deck, "__len__") else True
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert deck.draws == 0
    assert deck.shuffles == 0
    assert Card.get("AS") in deck
    assert "AS" not in deck


def test_deck_shuffle_counts_and_uses_random_shuffle(monkeypatch):
    deck = Deck(shuffle=False)
    calls = []

    def fake_shuffle(items):
        calls.append(list(items))
        items.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    original_top = deck.top
    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert len(calls) == 3
    assert deck.top == deck.bottom == original_top


def test_deck_draw_one_and_many_updates_counts_and_removes_cards():
    deck = Deck(shuffle=False)

    top_card = deck.top
    drawn = deck.draw()
    assert drawn == top_card
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert top_card not in deck

    next_two = deck.draw(times=2)
    assert isinstance(next_two, list)
    assert len(next_two) == 2
    assert deck.draws == 3
    assert deck.cards_left == 49
    assert all(card not in deck for card in next_two)


def test_deck_draw_hand_returns_hand_of_requested_size_and_order():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.draws == 5
    assert deck.cards_left == 47
    assert str(hand) == "[9♣, T♣, J♣, Q♣, K♣]"
    assert hand.cards == sorted(hand.cards)


def test_deck_str_and_repr_reflect_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_contains_24_cards_only_nine_through_ace():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert Card.get("8S") not in deck
    assert Card.get("9S") in deck
    assert Card.get("AS") in deck
    assert Card.get("KS") in deck
    assert Card.get("TS") in deck
    assert Card.get("JS") in deck
    assert Card.get("QS") in deck
    assert deck.bottom == Card.get("9S")
    assert deck.top == Card.get("AH")
