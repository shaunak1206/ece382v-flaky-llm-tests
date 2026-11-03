import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_suits_all_sorted_and_dict_keys():
    suits = Card.Suits.all()
    assert [s.value for s in suits] == [0, 1, 2, 3]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]
    suit_dict = Card.Suits.dict()
    assert set(suit_dict.keys()) == {"S", "C", "D", "H"}
    assert suit_dict["S"] is Card.Suits.SPADES
    assert suit_dict["H"] is Card.Suits.HEARTS


def test_values_all_sorted_and_dict_keys():
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
def test_card_get_parses_valid_cards_case_insensitively(text, expected_value, expected_suit):
    card = Card.get(text)
    assert card.value is expected_value
    assert card.suit is expected_suit
    assert str(card) == f"{expected_value.char}{expected_suit.symbol}"
    assert repr(card) == f"<Card:{expected_value.char}{expected_suit.symbol}>"


@pytest.mark.parametrize("text", ["A", "10S", "", "ASD", "AS "])
def test_card_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get(text)


@pytest.mark.parametrize("text, message", [("1S", "Invalid value for card!"), ("AZ", "Invalid suit for card!")])
def test_card_get_rejects_invalid_value_or_suit(text, message):
    with pytest.raises(InvalidCard, match=message):
        Card.get(text)


def test_card_properties_color_and_unicode_and_flags():
    spade_ace = Card.get("AS")
    heart_queen = Card.get("QH")

    assert spade_ace.color == Card.BLACK
    assert spade_ace.is_black is True
    assert spade_ace.is_red is False
    assert spade_ace.full_name == "ace of spades"
    assert spade_ace.unicode == chr(Card.Suits.SPADES.unicode + 1)

    assert heart_queen.color == Card.RED
    assert heart_queen.is_red is True
    assert heart_queen.is_black is False
    assert heart_queen.full_name == "queen of hearts"
    # queen/king use the "skip knight" offset
    assert heart_queen.unicode == chr(Card.Suits.HEARTS.unicode + 12 + 1)


def test_card_equality_and_ordering():
    ace_spades = Card.get("AS")
    ace_hearts = Card.get("AH")
    king_spades = Card.get("KS")

    assert ace_spades == Card.get("AS")
    assert ace_spades != ace_hearts
    assert ace_spades != "AS"
    assert ace_spades < king_spades
    assert king_spades > ace_spades
    assert ace_spades <= Card.get("AS")
    assert king_spades >= ace_spades


def test_hand_get_str_repr_and_len_sorted_cards():
    hand = Hand.get("KS, 2H, AD")
    assert len(hand) == 3
    assert [str(c) for c in hand.cards] == ["2♥", "A♦", "K♠"]
    assert str(hand) == "[2♥, A♦, K♠]"
    assert repr(hand) == "<Hand[3](high card) [2♥, A♦, K♠]>"


def test_hand_high_card_rank():
    hand = Hand.get("AS, 3D, 5C, 7H, 9S")
    assert hand.rank is Hand.Ranks.HIGH_CARD
    assert hand.is_straight is False
    assert hand.is_flush is False


def test_hand_pair_two_pair_three_of_a_kind_full_house_and_four_kind():
    pair = Hand.get("AS, AD, 5C, 7H, 9S")
    two_pair = Hand.get("AS, AD, 5C, 5H, 9S")
    trips = Hand.get("AS, AD, AC, 7H, 9S")
    full_house = Hand.get("AS, AD, AC, 7H, 7S")
    quads = Hand.get("AS, AD, AC, AH, 9S")

    assert pair.is_one_pair is True
    assert pair.rank is Hand.Ranks.PAIR

    assert two_pair.is_two_pair is True
    assert two_pair.rank is Hand.Ranks.TWO_PAIR

    assert trips.is_three_of_a_kind is True
    assert trips.rank is Hand.Ranks.THREE_OF_A_KIND

    assert full_house.is_full_house is True
    assert full_house.rank is Hand.Ranks.FULL_HOUSE

    assert quads.is_four_of_a_kind is True
    assert quads.rank is Hand.Ranks.FOUR_OF_A_KIND


def test_hand_flush_and_straight_and_straight_flush():
    flush = Hand.get("2H, 5H, 7H, JH, KH")
    straight = Hand.get("2S, 3D, 4C, 5H, 6S")
    straight_flush = Hand.get("9C, TC, JC, QC, KC")

    assert flush.is_flush is True
    assert flush.rank is Hand.Ranks.FLUSH

    assert straight.is_straight is True
    assert straight.rank is Hand.Ranks.STRAIGHT

    assert straight_flush.is_straight_flush is True
    assert straight_flush.rank is Hand.Ranks.STRAIGHT_FLUSH


def test_hand_straight_handles_ace_low_and_ace_high():
    ace_low = Hand.get("AS, 2D, 3C, 4H, 5S")
    ace_high = Hand.get("TS, JD, QC, KH, AS")

    assert ace_low.is_straight is True
    assert ace_low.rank is Hand.Ranks.STRAIGHT

    assert ace_high.is_straight is True
    assert ace_high.rank is Hand.Ranks.STRAIGHT


def test_hand_comparison_uses_rank_only():
    low = Hand.get("AS, 3D, 5C, 7H, 9S")
    high = Hand.get("AS, AD, 5C, 7H, 9S")
    assert low < high
    assert high > low


def test_deck_default_constructor_builds_52_cards_without_shuffle(monkeypatch):
    called = []

    def fake_shuffle(cards):
        called.append(list(cards))

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=False)

    assert len(deck.cards) == 52
    assert deck.cards_left == 52
    assert deck.shuffles == 0
    assert deck.draws == 0
    assert called == []
    assert deck.bottom == Card.get("AS")
    assert deck.top == Card.get("KH")
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"


def test_deck_shuffle_updates_counter_and_uses_random_shuffle(monkeypatch):
    calls = []

    def fake_shuffle(cards):
        calls.append(tuple(str(c) for c in cards))
        cards.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=False)
    original_top = deck.top
    deck.shuffle(times=2)

    assert deck.shuffles == 2
    assert len(calls) == 2
    assert deck.top == original_top
    assert deck.cards_left == 52


def test_deck_contains_only_cards():
    deck = Deck(shuffle=False)
    card = Card.get("AS")
    assert card in deck
    assert "AS" not in deck
    assert 123 not in deck


def test_deck_draw_and_draw_hand_reduce_counts():
    deck = Deck(cards=[Card.get("AS"), Card.get("2S"), Card.get("3S")], shuffle=False)

    one = deck.draw()
    assert one == Card.get("3S")
    assert deck.draws == 1
    assert deck.cards_left == 2

    many = deck.draw(times=2)
    assert many == [Card.get("2S"), Card.get("AS")]
    assert deck.draws == 3
    assert deck.cards_left == 0

    deck = Deck(cards=[Card.get("AS"), Card.get("2S"), Card.get("3S"), Card.get("4S"), Card.get("5S")], shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 0


def test_euchre_deck_contains_24_cards_and_expected_values():
    deck = EuchreDeck()
    assert len(deck.cards) == 24
    assert deck.cards_left == 24
    assert deck.draws == 0
    assert deck.shuffles == 0

    values = {c.value.value for c in deck.cards}
    assert values == {1, 9, 10, 11, 12, 13}

    assert Card.get("8S") not in deck
    assert Card.get("9S") in deck
    assert Card.get("AS") in deck
