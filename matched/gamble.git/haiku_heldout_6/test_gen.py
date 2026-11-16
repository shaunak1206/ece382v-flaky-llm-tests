import random

import pytest

from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES
    assert str(card) == "A\u2660"


def test_get_parses_valid_text():
    card = Card.get("TH")
    assert card.value == Card.Values.TEN
    assert card.suit == Card.Suits.HEARTS


def test_get_is_case_insensitive():
    assert Card.get("as") == Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)


@pytest.mark.parametrize("text", ["", "A", "AS1", "10S"])
def test_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_get_rejects_invalid_value_character():
    with pytest.raises(InvalidCard):
        Card.get("XS")


def test_get_rejects_invalid_suit_character():
    with pytest.raises(InvalidCard):
        Card.get("AX")


@pytest.mark.parametrize(
    "text, color_attr",
    [("AS", "is_black"), ("AC", "is_black"), ("AD", "is_red"), ("AH", "is_red")],
)
def test_color_properties_match_suit(text, color_attr):
    card = Card.get(text)
    assert getattr(card, color_attr) is True
    other = "is_red" if color_attr == "is_black" else "is_black"
    assert getattr(card, other) is False


def test_color_returns_enum_int():
    assert Card.get("KS").color == Card.BLACK
    assert Card.get("KH").color == Card.RED


def test_full_name():
    assert Card.get("TH").full_name == "ten of hearts"
    assert Card.get("AS").full_name == "ace of spades"


def test_str_and_repr():
    card = Card.get("QD")
    assert str(card) == "Q\u2666"
    assert repr(card) == "<Card:Q\u2666>"


def test_unicode_ace_of_spades_and_queen_skips_knight():
    assert Card.get("AS").unicode == "\U0001F0A1"
    # Queens and kings skip the knight code point
    assert Card.get("QS").unicode == "\U0001F0AD"
    assert Card.get("KS").unicode == "\U0001F0AE"


def test_ordering_uses_value_only():
    two_hearts = Card.get("2H")
    three_spades = Card.get("3S")
    assert two_hearts < three_spades
    assert three_spades > two_hearts
    assert Card.get("AS") < Card.get("2S")


def test_le_and_ge_with_lower_and_equal_cards():
    low = Card.get("2C")
    high = Card.get("9D")
    assert low <= high
    assert high >= low
    assert low <= Card.get("2C")
    assert high >= Card.get("9D")
    assert not (high <= low)
    assert not (low >= high)


def test_equality_requires_same_value_and_suit():
    assert Card.get("5H") == Card.get("5h")
    assert Card.get("5H") != Card.get("5S")
    assert Card.get("5H") != Card.get("6H")


def test_equality_with_non_card_is_false():
    assert (Card.get("5H") == "5H") is False
    assert (Card.get("5H") == None) is False  # noqa: E711


def test_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]
    assert [s.char for s in suits] == ["S", "C", "D", "H"]


def test_values_all_sorted_ascending_and_complete():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))


def test_values_dict_maps_characters():
    mapping = Card.Values.dict()
    assert mapping["A"] == Card.Values.ACE
    assert mapping["T"] == Card.Values.TEN
    assert mapping["K"] == Card.Values.KING
    assert len(mapping) == 13


def test_suits_dict_maps_characters():
    mapping = Card.Suits.dict()
    assert set(mapping) == {"S", "C", "D", "H"}
    assert mapping["H"] == Card.Suits.HEARTS


def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len({str(c) for c in deck.cards}) == 52


def test_unshuffled_deck_orders_top_ace_of_spades_bottom_king_of_hearts():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")
    assert deck.shuffles == 0


def test_constructor_shuffles_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "shuffle", lambda seq: calls.append(len(seq)))
    deck = Deck()
    assert deck.shuffles == 1
    assert calls == [52]


def test_draw_single_card_returns_top_and_updates_counters():
    deck = Deck(shuffle=False)
    expected_top = deck.top
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert drawn == expected_top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_returns_list_in_order():
    deck = Deck(shuffle=False)
    expected = [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    drawn = deck.draw(times=3)
    assert drawn == expected
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises():
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


def test_draw_hand_of_one_card_is_wrapped():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert len(hand) == 1
    assert deck.cards_left == 51


def test_shuffle_repeated_increments_counter_and_preserves_cards(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "shuffle", lambda seq: calls.append(True))
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    assert len(calls) == 3
    assert sorted(str(c) for c in deck.cards) == before


def test_contains_card_and_rejects_non_card():
    deck = Deck(shuffle=False)
    assert Card.get("KD") in deck
    assert "KD" not in deck


def test_contains_is_false_after_card_is_drawn():
    deck = Deck(shuffle=False)
    top = deck.draw()
    assert top not in deck


def test_custom_cards_are_used_without_shuffle():
    cards = [Card.get("2C"), Card.get("3C")]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == Card.get("3C")
    assert deck.bottom == Card.get("2C")


def test_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == "<Deck[52]>"
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_euchre_deck_has_24_cards_of_allowed_values():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    allowed = {9, 10, 11, 12, 13, 1}
    assert all(c.value.value in allowed for c in deck.cards)
