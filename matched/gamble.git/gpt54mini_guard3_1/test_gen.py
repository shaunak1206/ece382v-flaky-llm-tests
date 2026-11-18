import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suit_and_value_enums_are_sorted_and_complete():
    suits = Card.Suits.all()
    values = Card.Values.all()

    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    assert [s.value for s in suits] == [0, 1, 2, 3]

    assert [v.name for v in values] == [
        "ace",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "jack",
        "queen",
        "king",
    ]
    assert [v.char for v in values] == ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    assert [v.value for v in values] == list(range(1, 14))


def test_suit_and_value_dicts_map_characters_to_objects():
    suit_dict = Card.Suits.dict()
    value_dict = Card.Values.dict()

    assert set(suit_dict.keys()) == {"S", "C", "D", "H"}
    assert set(value_dict.keys()) == {"A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"}
    assert suit_dict["S"] is Card.Suits.SPADES
    assert suit_dict["H"] is Card.Suits.HEARTS
    assert value_dict["A"] is Card.Values.ACE
    assert value_dict["K"] is Card.Values.KING


def test_card_defaults_to_ace_of_spades():
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
    assert card.full_name == "queen of hearts"


def test_card_get_rejects_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("ASX")


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1S")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AZ")


def test_card_color_and_boolean_properties_match_suit():
    black_card = Card.get("AS")
    red_card = Card.get("AD")

    assert black_card.color == Card.BLACK
    assert black_card.is_black is True
    assert black_card.is_red is False

    assert red_card.color == Card.RED
    assert red_card.is_red is True
    assert red_card.is_black is False


def test_card_unicode_matches_expected_symbols():
    assert Card.get("AS").unicode == "🂡"
    assert Card.get("TS").unicode == "🂺"
    assert Card.get("QS").unicode == "🂭"
    assert Card.get("KS").unicode == "🂮"
    assert Card.get("AH").unicode == "🂱"
    assert Card.get("QH").unicode == "🂽"


def test_card_ordering_compares_by_value_only():
    low = Card.get("2S")
    high = Card.get("KD")
    same_value_different_suit = Card.get("KH")

    assert low < high
    assert high > low
    assert low <= low
    assert high >= high
    assert high == same_value_different_suit
    assert not (high != same_value_different_suit)
    assert low != high


def test_card_equality_against_non_card_is_false():
    assert Card.get("AS") != "AS"
    assert Card.get("AS") != object()


def test_deck_default_construction_has_52_unique_cards_and_known_top_bottom():
    deck = Deck(shuffle=False)

    assert deck.cards_left == 52
    assert deck.draws == 0
    assert deck.shuffles == 0
    assert deck.bottom == Card.get("KS")
    assert deck.top == Card.get("AS")

    unique_cards = {(card.value.value, card.suit.value) for card in deck.cards}
    assert len(unique_cards) == 52
    assert Card.get("AS") in deck
    assert "AS" not in deck


def test_deck_str_and_repr_reflect_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_deck_draw_one_updates_state_and_returns_top_card():
    deck = Deck(shuffle=False)
    card = deck.draw()

    assert card == Card.get("AS")
    assert deck.draws == 1
    assert deck.cards_left == 51
    assert deck.top == Card.get("KS")


def test_deck_draw_multiple_returns_list_and_updates_draw_count():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)

    assert isinstance(cards, list)
    assert cards == [Card.get("AS"), Card.get("KS"), Card.get("QS")]
    assert deck.draws == 3
    assert deck.cards_left == 49


def test_deck_draw_hand_returns_hand_with_sorted_cards():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert [str(card) for card in hand.cards] == ["9♠", "T♠", "J♠", "Q♠", "K♠"]
    assert hand.rank == Hand.Ranks.STRAIGHT_FLUSH
    assert deck.draws == 5
    assert deck.cards_left == 47


def test_deck_shuffle_uses_random_shuffle_and_counts_calls(monkeypatch):
    deck = Deck(shuffle=False)
    original = list(deck.cards)

    calls = []

    def fake_shuffle(items):
        calls.append(list(items))
        items.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)

    deck.shuffle(times=2)

    assert deck.shuffles == 2
    assert len(calls) == 2
    assert deck.cards == original
    assert deck.top == original[-1]
    assert deck.bottom == original[0]


def test_deck_shuffle_is_deterministic_with_seed():
    random.seed(12345)
    deck1 = Deck(shuffle=True)
    order1 = [str(card) for card in deck1.cards]

    random.seed(12345)
    deck2 = Deck(shuffle=True)
    order2 = [str(card) for card in deck2.cards]

    assert order1 == order2
    assert deck1.shuffles == 1
    assert deck2.shuffles == 1


def test_euchre_deck_contains_only_expected_values_and_24_cards():
    deck = EuchreDeck(shuffle=False)

    assert deck.cards_left == 24
    assert deck.shuffles == 0
    assert deck.draws == 0

    allowed = {1, 9, 10, 11, 12, 13}
    assert {card.value.value for card in deck.cards} == allowed
    assert {card.suit.value for card in deck.cards} == {0, 1, 2, 3}
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("9S")
