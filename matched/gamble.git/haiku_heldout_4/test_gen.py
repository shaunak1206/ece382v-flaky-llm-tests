import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


# ---------------------------------------------------------------------------
# Card construction and text representation
# ---------------------------------------------------------------------------


def test_card_default_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_str_uses_value_char_and_suit_symbol():
    assert str(Card(Card.Values.TEN, Card.Suits.HEARTS)) == "T\u2665"


def test_card_repr_wraps_str():
    card = Card(Card.Values.KING, Card.Suits.DIAMONDS)
    assert repr(card) == "<Card:K\u2666>"


def test_card_full_name():
    assert Card(Card.Values.ACE, Card.Suits.SPADES).full_name == "ace of spades"
    assert Card(Card.Values.TEN, Card.Suits.HEARTS).full_name == "ten of hearts"


# ---------------------------------------------------------------------------
# Card.get parsing
# ---------------------------------------------------------------------------


def test_card_get_parses_valid_string():
    card = Card.get("AS")
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_get_is_case_insensitive():
    assert Card.get("ts") == Card(Card.Values.TEN, Card.Suits.SPADES)


@pytest.mark.parametrize("text", ["", "A", "AS2", "10S"])
def test_card_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_get_rejects_invalid_value():
    with pytest.raises(InvalidCard):
        Card.get("1S")


def test_card_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard):
        Card.get("AX")


# ---------------------------------------------------------------------------
# Card color and unicode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suit", [Card.Suits.SPADES, Card.Suits.CLUBS])
def test_black_suits_are_black(suit):
    card = Card(Card.Values.FIVE, suit)
    assert card.color == Card.BLACK
    assert card.is_black is True
    assert card.is_red is False


@pytest.mark.parametrize("suit", [Card.Suits.DIAMONDS, Card.Suits.HEARTS])
def test_red_suits_are_red(suit):
    card = Card(Card.Values.FIVE, suit)
    assert card.color == Card.RED
    assert card.is_red is True
    assert card.is_black is False


def test_unicode_for_ace_of_spades():
    assert Card(Card.Values.ACE, Card.Suits.SPADES).unicode == "\U0001F0A1"


def test_unicode_skips_knight_for_queen_and_king():
    assert Card(Card.Values.QUEEN, Card.Suits.SPADES).unicode == "\U0001F0AD"
    assert Card(Card.Values.KING, Card.Suits.SPADES).unicode == "\U0001F0AE"


# ---------------------------------------------------------------------------
# Card equality and ordering
# ---------------------------------------------------------------------------


def test_cards_with_same_value_and_suit_are_equal():
    assert Card.get("KH") == Card.get("kh")


def test_cards_with_different_suit_are_not_equal():
    assert Card.get("KH") != Card.get("KS")


def test_card_not_equal_to_non_card():
    assert (Card() == "A\u2660") is False


def test_card_ordering_uses_value():
    low = Card(Card.Values.TWO, Card.Suits.HEARTS)
    high = Card(Card.Values.KING, Card.Suits.SPADES)
    assert low < high
    assert high > low
    assert not (high < low)


def test_card_le_and_ge_on_equal_cards():
    card = Card.get("QD")
    same = Card.get("QD")
    assert card <= same
    assert card >= same


def test_card_le_and_ge_on_different_values():
    ace = Card.get("AC")
    king = Card.get("KC")
    assert ace <= king
    assert king >= ace
    assert not (king <= ace)
    assert not (ace >= king)


# ---------------------------------------------------------------------------
# Deck construction and ordering
# ---------------------------------------------------------------------------


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    unique = {(c.value.char, c.suit.char) for c in deck.cards}
    assert len(unique) == 52


def test_unshuffled_deck_top_is_ace_of_spades_and_bottom_is_king_of_hearts():
    deck = Deck(shuffle=False)
    assert deck.top == Card(Card.Values.ACE, Card.Suits.SPADES)
    assert deck.bottom == Card(Card.Values.KING, Card.Suits.HEARTS)


def test_deck_constructed_with_cards_uses_those_cards():
    cards = [Card.get("2C"), Card.get("3D"), Card.get("4H")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 3
    assert deck.top == Card.get("4H")
    assert deck.bottom == Card.get("2C")


def test_deck_shuffle_true_records_one_shuffle():
    deck = Deck()
    assert deck.shuffles == 1
    assert deck.draws == 0


def test_deck_shuffle_false_records_no_shuffles():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0


def test_deck_shuffle_calls_random_shuffle(monkeypatch):
    calls = []

    def fake_shuffle(seq):
        calls.append(seq)

    monkeypatch.setattr(random, "shuffle", fake_shuffle)
    deck = Deck(shuffle=False)
    deck.shuffle(times=2)
    assert len(calls) == 2
    assert deck.shuffles == 2
    assert all(seq is deck.cards for seq in calls)


# ---------------------------------------------------------------------------
# Deck drawing and membership
# ---------------------------------------------------------------------------


def test_draw_single_card_returns_top_and_removes_it():
    deck = Deck(shuffle=False)
    expected_top = deck.top
    drawn = deck.draw()
    assert drawn == expected_top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_returns_list_in_order():
    deck = Deck(shuffle=False)
    expected = [deck.cards[-1], deck.cards[-2], deck.cards[-3]]
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == expected
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises_index_error():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 47


def test_draw_hand_of_size_one_still_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert len(hand) == 1


def test_contains_card_in_deck_and_after_draw():
    deck = Deck(shuffle=False)
    top = deck.top
    assert top in deck
    deck.draw()
    assert top not in deck


def test_contains_non_card_returns_false():
    deck = Deck(shuffle=False)
    assert "AS" not in deck
    assert None not in deck


def test_deck_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"
    assert repr(deck) == str(deck)


# ---------------------------------------------------------------------------
# EuchreDeck
# ---------------------------------------------------------------------------


def test_euchre_deck_has_24_cards_with_only_euchre_values():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    allowed = {1, 9, 10, 11, 12, 13}
    assert all(card.value.value in allowed for card in deck.cards)
    assert len({(c.value.char, c.suit.char) for c in deck.cards}) == 24
