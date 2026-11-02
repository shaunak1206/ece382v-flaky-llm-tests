import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def _key(card):
    return (card.value.char, card.suit.char)


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_get_parses_value_and_suit_case_insensitively():
    assert Card.get("AS").value is Card.Values.ACE
    assert Card.get("AS").suit is Card.Suits.SPADES
    assert Card.get("th").value is Card.Values.TEN
    assert Card.get("th").suit is Card.Suits.HEARTS
    assert Card.get("9d") == Card(value=Card.Values.NINE, suit=Card.Suits.DIAMONDS)


def test_get_rejects_wrong_length():
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("ASS")


def test_get_rejects_unknown_value():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("ZH")


def test_get_rejects_unknown_suit():
    with pytest.raises(InvalidCard):
        Card.get("AX")
    with pytest.raises(InvalidCard):
        Card.get("A1")


def test_str_and_repr():
    card = Card.get("AS")
    assert str(card) == "A♠"
    assert repr(card) == "<Card:A♠>"
    assert str(Card.get("TH")) == "T♥"


def test_full_name():
    assert Card.get("AS").full_name == "ace of spades"
    assert Card.get("TH").full_name == "ten of hearts"
    assert (
        Card(value=Card.Values.QUEEN, suit=Card.Suits.CLUBS).full_name
        == "queen of clubs"
    )


def test_color_flags_for_each_suit():
    for suit in (Card.Suits.SPADES, Card.Suits.CLUBS):
        card = Card(value=Card.Values.KING, suit=suit)
        assert card.color == Card.BLACK
        assert card.is_black
        assert not card.is_red
    for suit in (Card.Suits.DIAMONDS, Card.Suits.HEARTS):
        card = Card(value=Card.Values.KING, suit=suit)
        assert card.color == Card.RED
        assert card.is_red
        assert not card.is_black


def test_unicode_skips_knight_for_queen_and_king():
    assert Card.get("AS").unicode == "\U0001F0A1"
    assert Card.get("TS").unicode == "\U0001F0AA"
    assert Card.get("JS").unicode == "\U0001F0AB"
    assert Card.get("QS").unicode == "\U0001F0AD"
    assert Card.get("KS").unicode == "\U0001F0AE"
    assert Card.get("KH").unicode == "\U0001F0BE"


def test_equality_compares_value_and_suit():
    assert Card.get("QD") == Card(value=Card.Values.QUEEN, suit=Card.Suits.DIAMONDS)
    assert Card.get("qd") == Card.get("QD")
    assert Card.get("QD") != Card.get("QH")
    assert Card.get("QD") != Card.get("KD")
    assert Card.get("QD") != "QD"


def test_ordering_uses_value_with_ace_low():
    ace = Card.get("AH")
    two = Card.get("2S")
    three = Card.get("3H")
    ten = Card.get("TC")
    king = Card.get("KD")
    assert ace < two < three < ten < king
    assert king > ten > three > two > ace
    assert ten <= king and king >= ten
    assert ten <= ten and ten >= ten


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(deck.cards) == 52
    assert len({_key(c) for c in deck.cards}) == 52


def test_unshuffled_deck_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_shuffle_flag_and_shuffle_counter():
    assert Deck(shuffle=False).shuffles == 0
    assert Deck(shuffle=False).draws == 0
    assert Deck().shuffles == 1
    assert Deck(shuffle=True).shuffles == 1


def test_shuffle_calls_random_shuffle_each_time(monkeypatch):
    calls = []

    def fake_shuffle(seq):
        calls.append(len(seq))
        seq.reverse()

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    a, b, c = Card.get("2S"), Card.get("3S"), Card.get("4S")
    deck = Deck(cards=[a, b, c], shuffle=True)
    assert calls == [3]
    assert deck.shuffles == 1
    assert deck.cards == [c, b, a]

    deck.shuffle(times=2)
    assert calls == [3, 3, 3]
    assert deck.shuffles == 3
    assert deck.cards == [c, b, a]


def test_draw_single_returns_top_card_and_updates_counters():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert isinstance(card, Card)
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert deck.draw() == Card.get("2S")
    assert deck.draws == 2


def test_draw_multiple_returns_list_in_draw_order():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert [_key(c) for c in cards] == [("A", "S"), ("2", "S"), ("3", "S")]
    assert deck.cards_left == 49
    assert deck.draws == 3
    assert deck.top == Card.get("4S")


def test_draw_from_empty_deck_raises_index_error():
    deck = Deck(cards=[Card.get("2H"), Card.get("3H")], shuffle=False)
    assert deck.draw() == Card.get("3H")
    assert deck.draw() == Card.get("2H")
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_default_size_five():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand()
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert hand.size == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_draw_hand_size_one_wraps_single_card():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert len(hand) == 1
    assert hand.cards[0] == Card.get("AS")
    assert deck.cards_left == 51


def test_contains_and_str_reflect_cards_left():
    deck = Deck(shuffle=False)
    ace = Card.get("AS")
    assert ace in deck
    assert Card.get("KH") in deck
    assert "AS" not in deck
    deck.draw()
    assert ace not in deck
    assert str(deck) == "<Deck[51]>"
    assert repr(deck) == "<Deck[51]>"


def test_custom_cards_are_used_as_given():
    cards = [Card.get("2C"), Card.get("7D")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards == cards
    assert deck.cards_left == 2
    assert deck.top == Card.get("7D")
    assert deck.bottom == Card.get("2C")


def test_euchre_deck_contains_only_nine_through_ace():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert len({_key(c) for c in deck.cards}) == 24
    assert {c.value.value for c in deck.cards} == {1, 9, 10, 11, 12, 13}
    for suit in Card.Suits.all():
        assert sum(1 for c in deck.cards if c.suit is suit) == 6
    assert Card.get("9H") in deck
    assert Card.get("2H") not in deck


def test_euchre_deck_is_a_shuffled_deck():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.shuffles == 1
    assert deck.draws == 0
