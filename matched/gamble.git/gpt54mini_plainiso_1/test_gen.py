import random
from unittest.mock import patch

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
    assert [v.char for v in values[:4]] == ["A", "2", "3", "4"]
    assert values[-1].char == "K"


def test_values_dict_maps_characters_to_values():
    values = Card.Values.dict()
    assert values["A"] is Card.Values.ACE
    assert values["T"] is Card.Values.TEN
    assert values["K"] is Card.Values.KING


def test_card_defaults_are_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"


def test_card_get_parses_case_insensitive_text():
    card = Card.get("qh")
    assert card.value is Card.Values.QUEEN
    assert card.suit is Card.Suits.HEARTS
    assert str(card) == "Q♥"


def test_card_get_rejects_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("ASX")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1S")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AX")


def test_card_color_and_red_black_properties():
    spade = Card.get("AS")
    diamond = Card.get("AD")
    assert spade.color == Card.BLACK
    assert spade.is_black is True
    assert spade.is_red is False

    assert diamond.color == Card.RED
    assert diamond.is_red is True
    assert diamond.is_black is False


def test_card_unicode_special_cases_for_face_cards():
    assert Card.get("JS").unicode == "🂫"
    assert Card.get("QS").unicode == "🂭"
    assert Card.get("KS").unicode == "🂮"
    assert Card.get("AS").unicode == "🂡"


def test_card_comparisons_and_equality():
    two_spades = Card.get("2S")
    three_hearts = Card.get("3H")
    another_two_spades = Card.get("2S")

    assert two_spades == another_two_spades
    assert two_spades != three_hearts
    assert two_spades < three_hearts
    assert three_hearts > two_spades
    assert two_spades <= another_two_spades
    assert three_hearts >= two_spades
    assert (two_spades == "2S") is False


def test_deck_default_constructor_creates_52_cards_without_shuffle():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert deck.draws == 0
    assert deck.shuffles == 0
    assert isinstance(deck.top, Card)
    assert isinstance(deck.bottom, Card)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_default_order_top_and_bottom_are_expected():
    deck = Deck(shuffle=False)
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert deck.cards[0] == Card.get("AS")
    assert deck.cards[-1] == Card.get("KH")


def test_deck_contains_only_cards():
    deck = Deck(shuffle=False)
    assert Card.get("AS") in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_single_card_updates_counts_and_removes_top_card():
    deck = Deck(shuffle=False)
    top_before = deck.top

    drawn = deck.draw()

    assert drawn == top_before
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert drawn not in deck
    assert deck.top == Card.get("QH")


def test_deck_draw_multiple_cards_returns_list_in_draw_order():
    deck = Deck(shuffle=False)

    drawn = deck.draw(times=3)

    assert isinstance(drawn, list)
    assert len(drawn) == 3
    assert drawn == [Card.get("KH"), Card.get("QH"), Card.get("JH")]
    assert deck.draws == 3
    assert deck.cards_left == 49


def test_deck_draw_hand_returns_hand_with_expected_cards():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert hand.cards == [Card.get("TH"), Card.get("JH"), Card.get("QH"), Card.get("KH"), Card.get("AH")]
    assert deck.draws == 5
    assert deck.cards_left == 47


def test_deck_shuffle_calls_random_shuffle_requested_number_of_times():
    deck = Deck(shuffle=False)
    with patch("gamble.models.cards.random.shuffle") as mock_shuffle:
        deck.shuffle(times=3)

    assert deck.shuffles == 3
    assert mock_shuffle.call_count == 3
    assert mock_shuffle.call_args_list == [((deck.cards,),), ((deck.cards,),), ((deck.cards,),)]


def test_deck_init_with_shuffle_triggers_shuffle_once(monkeypatch):
    calls = []

    def fake_shuffle(cards):
        calls.append(list(cards))

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=True)

    assert deck.shuffles == 1
    assert len(calls) == 1
    assert deck.cards_left == 52


def test_euchre_deck_contains_24_cards_and_expected_bounds():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("9H")

    values = {card.value.value for card in deck.cards}
    assert values == {1, 9, 10, 11, 12, 13}
