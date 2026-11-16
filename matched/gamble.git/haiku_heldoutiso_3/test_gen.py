import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


# ---------------------------------------------------------------------------
# Card construction and parsing
# ---------------------------------------------------------------------------


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_stores_given_value_and_suit():
    card = Card(value=Card.Values.TEN, suit=Card.Suits.HEARTS)
    assert card.value is Card.Values.TEN
    assert card.suit is Card.Suits.HEARTS


def test_get_parses_valid_text_case_insensitively():
    assert Card.get("AS") == Card(Card.Values.ACE, Card.Suits.SPADES)
    assert Card.get("as") == Card(Card.Values.ACE, Card.Suits.SPADES)
    assert Card.get("td") == Card(Card.Values.TEN, Card.Suits.DIAMONDS)
    assert Card.get("KH").value is Card.Values.KING
    assert Card.get("KH").suit is Card.Suits.HEARTS


@pytest.mark.parametrize("text", ["", "A", "AS1", "10S", "ASS"])
def test_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


@pytest.mark.parametrize("text", ["1S", "XS", "0C"])
def test_get_rejects_invalid_value(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


@pytest.mark.parametrize("text", ["AX", "A1", "A?"])
def test_get_rejects_invalid_suit(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


# ---------------------------------------------------------------------------
# Card equality and ordering
# ---------------------------------------------------------------------------


def test_equality_requires_same_value_and_suit():
    assert Card.get("7H") == Card.get("7h")
    assert Card.get("7H") != Card.get("7S")
    assert Card.get("7H") != Card.get("8H")


def test_card_not_equal_to_non_card_objects():
    assert (Card.get("AS") == "AS") is False
    assert (Card.get("AS") != "AS") is True
    assert (Card.get("AS") == None) is False  # noqa: E711


def test_ordering_compares_values_and_ignores_suit():
    ace = Card.get("AS")
    two = Card.get("2H")
    king = Card.get("KD")
    ten_clubs = Card.get("TC")
    ten_spades = Card.get("TS")

    assert ace < two
    assert king > ten_clubs
    assert ten_clubs <= ten_spades
    assert ten_clubs >= ten_spades
    assert ten_clubs <= king
    assert king >= ace
    assert not (king < ace)


# ---------------------------------------------------------------------------
# Card properties and string representations
# ---------------------------------------------------------------------------


def test_color_and_color_helpers():
    spade = Card.get("9S")
    club = Card.get("9C")
    diamond = Card.get("9D")
    heart = Card.get("9H")

    assert spade.color == Card.BLACK
    assert club.is_black and not club.is_red
    assert diamond.color == Card.RED
    assert heart.is_red and not heart.is_black


def test_full_name():
    assert Card.get("AS").full_name == "ace of spades"
    assert Card.get("TH").full_name == "ten of hearts"


def test_str_and_repr():
    card = Card.get("QD")
    assert str(card) == "Q♦"
    assert repr(card) == "<Card:Q♦>"


def test_unicode_skips_knight_for_queen_and_king():
    # Ace of spades is U+1F0A1
    assert Card.get("AS").unicode == chr(0x1F0A1)
    # Queen of spades is U+1F0AD (the knight U+1F0AC is skipped)
    assert Card.get("QS").unicode == chr(0x1F0AD)
    # King of hearts is U+1F0BE
    assert Card.get("KH").unicode == chr(0x1F0BE)


# ---------------------------------------------------------------------------
# Enum-like namespaces
# ---------------------------------------------------------------------------


def test_suits_all_sorted_by_value():
    names = [suit.name for suit in Card.Suits.all()]
    assert names == ["spades", "clubs", "diamonds", "hearts"]


def test_values_all_sorted_and_complete():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING


def test_suits_and_values_dicts_keyed_by_char():
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}
    assert Card.Suits.dict()["H"] is Card.Suits.HEARTS
    assert Card.Values.dict()["T"] is Card.Values.TEN
    assert len(Card.Values.dict()) == 13


# ---------------------------------------------------------------------------
# Deck construction and state
# ---------------------------------------------------------------------------


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len({str(card) for card in deck.cards}) == 52


def test_unshuffled_deck_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_shuffle_counter_on_construction():
    assert Deck().shuffles == 1
    assert Deck(shuffle=False).shuffles == 0


def test_shuffle_times_increments_counter_and_keeps_cards():
    deck = Deck(shuffle=False)
    before = {str(card) for card in deck.cards}
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert {str(card) for card in deck.cards} == before


def test_shuffle_delegates_to_random_shuffle(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "shuffle", lambda seq: calls.append(list(seq)))
    deck = Deck(cards=[Card.get("AS"), Card.get("2S")], shuffle=False)
    deck.shuffle()
    assert calls == [[Card.get("AS"), Card.get("2S")]]


def test_custom_cards_are_used_as_given():
    cards = [Card.get("3C"), Card.get("4D")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("4D")
    assert deck.bottom == Card.get("3C")


# ---------------------------------------------------------------------------
# Deck drawing and membership
# ---------------------------------------------------------------------------


def test_draw_single_card_returns_top_and_updates_state():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert card == Card.get("AS")
    assert isinstance(card, Card)
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list_in_order():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert cards == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_exhausted_deck_raises():
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
    assert deck.draws == 5


def test_contains_checks_cards_and_ignores_other_types():
    deck = Deck(cards=[Card.get("AS"), Card.get("KD")], shuffle=False)
    assert Card.get("KD") in deck
    assert Card.get("2C") not in deck
    assert "AS" not in deck


# ---------------------------------------------------------------------------
# Deck string representation and subclasses
# ---------------------------------------------------------------------------


def test_deck_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_contains_only_nines_through_aces():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    assert all(card.value.value >= 9 or card.value.value == 1 for card in deck.cards)
    assert Card.get("9S") in deck
    assert Card.get("AH") in deck
    assert Card.get("8S") not in deck
    assert Card.get("2C") not in deck
    assert deck.shuffles == 1
