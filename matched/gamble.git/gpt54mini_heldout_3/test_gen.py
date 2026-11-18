import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, Hand


def test_suit_all_returns_sorted_suits():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_suit_dict_maps_characters_to_suits():
    suits = Card.Suits.dict()
    assert set(suits.keys()) == {"S", "C", "D", "H"}
    assert suits["S"] is Card.Suits.SPADES
    assert suits["H"] is Card.Suits.HEARTS


def test_values_all_returns_sorted_values():
    values = Card.Values.all()
    assert [v.value for v in values] == list(range(1, 14))
    assert [v.char for v in values] == ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def test_values_dict_maps_characters_to_values():
    values = Card.Values.dict()
    assert set(values.keys()) == set("A23456789TJQK")
    assert values["A"] is Card.Values.ACE
    assert values["K"] is Card.Values.KING


def test_card_default_constructor_uses_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_get_parses_valid_text_case_insensitively():
    card = Card.get("qh")
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "Q♥"


def test_card_get_raises_for_too_many_characters():
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("10S")


def test_card_get_raises_for_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("XS")


def test_card_get_raises_for_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AZ")


def test_card_color_and_boolean_properties():
    spade = Card.get("AS")
    heart = Card.get("AH")

    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False

    assert heart.color == Card.RED
    assert heart.is_black is False
    assert heart.is_red is True


def test_card_unicode_for_queen_and_king_skips_knight_gap():
    queen_spade = Card.get("QS")
    king_spade = Card.get("KS")
    jack_spade = Card.get("JS")

    assert jack_spade.unicode == chr(Card.Suits.SPADES.unicode + 11)
    assert queen_spade.unicode == chr(Card.Suits.SPADES.unicode + 12 + 1)
    assert king_spade.unicode == chr(Card.Suits.SPADES.unicode + 13 + 1)


def test_card_comparisons_and_equality():
    two_clubs = Card.get("2C")
    ten_hearts = Card.get("TH")
    another_ten_hearts = Card.get("TH")

    assert two_clubs < ten_hearts
    assert ten_hearts > two_clubs
    assert two_clubs <= ten_hearts
    assert ten_hearts >= two_clubs
    assert ten_hearts == another_ten_hearts
    assert ten_hearts != two_clubs
    assert (ten_hearts == "TH") is False


def test_deck_default_constructor_builds_full_deck_without_shuffle():
    deck = Deck(shuffle=False)

    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KS")
    assert Card.get("AS") in deck
    assert "AS" not in deck


def test_deck_draw_reduces_size_and_tracks_draws():
    deck = Deck(shuffle=False)
    top_card = deck.top

    drawn = deck.draw()

    assert isinstance(drawn, Card)
    assert drawn == top_card
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert drawn not in deck


def test_deck_draw_multiple_returns_list_and_preserves_order_from_top_down():
    deck = Deck(shuffle=False)
    top_before = deck.top
    second_before = deck.cards[-2]

    drawn = deck.draw(times=2)

    assert isinstance(drawn, list)
    assert drawn == [top_before, second_before]
    assert deck.cards_left == 50
    assert deck.draws == 2


def test_deck_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_deck_shuffle_tracks_shuffle_count_and_uses_random_shuffle(monkeypatch):
    deck = Deck(cards=[Card.get("AS"), Card.get("2S"), Card.get("3S")], shuffle=False)
    calls = []

    def fake_shuffle(seq):
        calls.append(list(seq))
        seq.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert len(calls) == 3
    assert deck.cards == [Card.get("AS"), Card.get("2S"), Card.get("3S")]


def test_deck_str_and_repr_reflect_cards_left():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S")], shuffle=False)
    assert str(deck) == "<Deck[2]>"
    assert repr(deck) == "<Deck[2]>"
