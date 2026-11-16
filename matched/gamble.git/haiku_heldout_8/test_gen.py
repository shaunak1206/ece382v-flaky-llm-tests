import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def make_card(value_name, suit_name="SPADES"):
    return Card(value=getattr(Card.Values, value_name), suit=getattr(Card.Suits, suit_name))


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_card_str_and_repr():
    card = make_card("ACE", "SPADES")
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"


def test_card_full_name():
    assert make_card("QUEEN", "HEARTS").full_name == "queen of hearts"


@pytest.mark.parametrize(
    "suit_name, expected_color, is_black, is_red",
    [
        ("SPADES", Card.BLACK, True, False),
        ("CLUBS", Card.BLACK, True, False),
        ("DIAMONDS", Card.RED, False, True),
        ("HEARTS", Card.RED, False, True),
    ],
)
def test_color_and_color_properties(suit_name, expected_color, is_black, is_red):
    card = make_card("FIVE", suit_name)
    assert card.color == expected_color
    assert card.is_black is is_black
    assert card.is_red is is_red


@pytest.mark.parametrize(
    "text, value_name, suit_name",
    [
        ("AS", "ACE", "SPADES"),
        ("TH", "TEN", "HEARTS"),
        ("2C", "TWO", "CLUBS"),
        ("KD", "KING", "DIAMONDS"),
    ],
)
def test_get_parses_valid_text(text, value_name, suit_name):
    card = Card.get(text)
    assert card.value is getattr(Card.Values, value_name)
    assert card.suit is getattr(Card.Suits, suit_name)


def test_get_is_case_insensitive():
    assert Card.get("as") == Card.get("AS")
    assert Card.get("th") == Card.get("TH")


@pytest.mark.parametrize("text", ["", "A", "10S", "ASS"])
def test_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_get_rejects_invalid_value():
    with pytest.raises(InvalidCard):
        Card.get("XS")


def test_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_unicode_skips_knight_for_queen_and_king():
    assert make_card("ACE", "SPADES").unicode == "\U0001F0A1"
    assert make_card("QUEEN", "SPADES").unicode == "\U0001F0AD"
    assert make_card("KING", "SPADES").unicode == "\U0001F0AE"


def test_card_ordering_by_value_including_inclusive_comparisons():
    two = make_card("TWO")
    five = make_card("FIVE", "HEARTS")
    five_again = make_card("FIVE", "SPADES")

    assert two < five
    assert five > two
    assert two <= five
    assert five >= two
    assert five <= five_again
    assert five >= five_again
    assert not five < five_again
    assert not five > five_again


def test_card_equality_depends_on_value_and_suit():
    assert make_card("KING", "HEARTS") == Card.get("KH")
    assert make_card("KING", "HEARTS") != make_card("KING", "SPADES")
    assert make_card("KING", "HEARTS") != make_card("QUEEN", "HEARTS")
    assert Card() != "AS"
    assert (Card() == 42) is False


def test_suits_all_is_sorted_and_complete():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert Card.Suits.dict() == {s.char: s for s in suits}


def test_values_all_is_sorted_and_complete():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert set(Card.Values.dict().keys()) == {
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"
    }


def test_default_deck_is_full_unique_52_cards():
    deck = Deck()
    assert deck.cards_left == 52
    assert len({str(card) for card in deck.cards}) == 52


def test_unshuffled_deck_has_expected_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.shuffles == 0
    assert deck.top == make_card("ACE", "SPADES")
    assert deck.bottom == make_card("KING", "HEARTS")


def test_deck_constructor_shuffles_by_default_and_counts_it():
    assert Deck().shuffles == 1
    assert Deck(shuffle=False).shuffles == 0


def test_shuffle_increments_counter_and_keeps_same_cards():
    deck = Deck(shuffle=False)
    original = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert sorted(str(c) for c in deck.cards) == original


def test_seeded_shuffle_is_reproducible():
    random.seed(1234)
    first = Deck(shuffle=False)
    first.shuffle()

    random.seed(1234)
    second = Deck(shuffle=False)
    second.shuffle()

    assert first.cards == second.cards


def test_draw_single_card_removes_top_and_counts():
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert drawn == top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_returns_list_from_top():
    deck = Deck(shuffle=False)
    expected = [deck.cards[-1], deck.cards[-2], deck.cards[-3]]
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == expected
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises():
    deck = Deck(cards=[Card()], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck()
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert hand.size == 5
    assert deck.cards_left == 47


def test_draw_hand_with_size_one_wraps_single_card():
    deck = Deck()
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert hand.size == 1
    assert deck.cards_left == 51


def test_contains_checks_cards_and_ignores_non_cards():
    deck = Deck(shuffle=False)
    ace = make_card("ACE", "SPADES")
    assert ace in deck
    assert "AS" not in deck
    deck.draw(times=52)
    assert ace not in deck


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_has_only_nines_through_aces():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    assert len({str(card) for card in deck.cards}) == 24
    for card in deck.cards:
        assert card.value.value == 1 or card.value.value >= 9
    assert deck.shuffles == 1
