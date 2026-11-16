"""
Unit tests for Card and Deck in gamble/models/cards.py.
"""
import gamble.models.cards as cards_module
import pytest
from gamble.errors import InvalidCard
from gamble.models.cards import Card, Deck, EuchreDeck, Hand


def all_card_strings(deck):
    return sorted(str(card) for card in deck.cards)


def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES
    assert card == Card.get("AS")


def test_card_str_repr_and_full_name():
    card = Card.get("AS")
    assert str(card) == "A\u2660"
    assert repr(card) == "<Card:A\u2660>"
    assert card.full_name == "ace of spades"
    assert Card.get("TH").full_name == "ten of hearts"


@pytest.mark.parametrize(
    "text, value_char, suit_name",
    [
        ("2C", "2", "clubs"),
        ("TD", "T", "diamonds"),
        ("KH", "K", "hearts"),
        ("QS", "Q", "spades"),
    ],
)
def test_get_parses_valid_text(text, value_char, suit_name):
    card = Card.get(text)
    assert card.value.char == value_char
    assert card.suit.name == suit_name


def test_get_is_case_insensitive():
    assert Card.get("ah") == Card.get("AH")
    assert Card.get("tc") == Card.get("TC")


def test_get_rejects_wrong_length():
    with pytest.raises(InvalidCard):
        Card.get("10S")
    with pytest.raises(InvalidCard):
        Card.get("A")


def test_get_rejects_invalid_value_and_suit():
    with pytest.raises(InvalidCard):
        Card.get("1S")
    with pytest.raises(InvalidCard):
        Card.get("AX")


def test_color_is_red_and_is_black():
    spades = Card.get("AS")
    hearts = Card.get("AH")
    assert spades.color == Card.BLACK
    assert spades.is_black is True
    assert spades.is_red is False
    assert hearts.color == Card.RED
    assert hearts.is_red is True
    assert hearts.is_black is False


def test_unicode_for_regular_card():
    assert Card.get("2H").unicode == chr(127152 + 2)
    assert Card.get("JS").unicode == chr(127136 + 11)


def test_unicode_skips_knight_for_queen_and_king():
    assert Card.get("QS").unicode == chr(127136 + 12 + 1)
    assert Card.get("KS").unicode == chr(127136 + 13 + 1)


def test_ordering_comparisons_use_value():
    ace = Card.get("AS")
    two = Card.get("2H")
    king = Card.get("KD")
    assert ace < two
    assert king > two
    assert not king < two
    assert not ace > two


def test_ordering_less_or_equal_and_greater_or_equal():
    five_s = Card.get("5S")
    five_h = Card.get("5H")
    six = Card.get("6C")
    assert five_s <= five_h
    assert five_s >= five_h
    assert five_s <= six
    assert six >= five_s
    assert not six <= five_s
    assert not five_s >= six


def test_equality_requires_same_value_and_suit():
    assert Card.get("7D") == Card.get("7D")
    assert Card.get("7D") != Card.get("7H")
    assert Card.get("7D") != Card.get("8D")


def test_equality_with_non_card_is_false():
    assert (Card.get("AS") == "AS") is False
    assert (Card.get("AS") == None) is False  # noqa: E711


def test_suits_all_sorted_by_value():
    names = [suit.name for suit in Card.Suits.all()]
    assert names == ["spades", "clubs", "diamonds", "hearts"]
    assert set(Card.Suits.dict().keys()) == {"S", "C", "D", "H"}


def test_values_all_sorted_with_ace_low():
    values = Card.Values.all()
    assert len(values) == 13
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING
    assert [v.value for v in values] == list(range(1, 14))
    assert Card.Values.dict()["T"] is Card.Values.TEN


def test_deck_default_has_52_unique_cards_when_unshuffled():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len(set(str(card) for card in deck.cards)) == 52
    assert deck.shuffles == 0


def test_deck_unshuffled_top_and_bottom():
    deck = Deck(shuffle=False)
    assert deck.top == Card.get("AS")
    assert deck.bottom == Card.get("KH")


def test_deck_shuffle_flag_increments_shuffles():
    assert Deck(shuffle=True).shuffles == 1
    assert Deck(shuffle=False).shuffles == 0


def test_shuffle_calls_random_shuffle_and_counts(monkeypatch):
    deck = Deck(shuffle=False)
    before = list(deck.cards)
    monkeypatch.setattr(cards_module.random, "shuffle", lambda seq: seq.reverse())
    deck.shuffle()
    assert deck.cards == list(reversed(before))
    assert deck.shuffles == 1
    deck.shuffle(times=3)
    assert deck.shuffles == 4


def test_draw_single_card_removes_top():
    deck = Deck(shuffle=False)
    card = deck.draw()
    assert card == Card.get("AS")
    assert deck.cards_left == 51
    assert deck.draws == 1
    assert card not in deck


def test_draw_multiple_cards_returns_list_in_order():
    deck = Deck(shuffle=False)
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert cards == [Card.get("AS"), Card.get("2S"), Card.get("3S")]
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert hand.size == 5
    assert len(hand) == 5
    assert deck.cards_left == 47


def test_draw_hand_of_one_card_still_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=1)
    assert isinstance(hand, Hand)
    assert len(hand) == 1
    assert hand.cards[0] == Card.get("AS")


def test_contains_checks_cards_and_ignores_other_types():
    deck = Deck(shuffle=False)
    assert Card.get("KH") in deck
    assert "AS" not in deck
    deck.draw()
    assert Card.get("AS") not in deck


def test_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    assert str(deck) == "<Deck[52]>"
    assert repr(deck) == str(deck)
    deck.draw()
    assert str(deck) == "<Deck[51]>"


def test_draw_from_empty_deck_raises_index_error():
    deck = Deck(cards=[Card.get("AS")], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_euchre_deck_has_24_cards_of_nine_through_ace():
    deck = EuchreDeck()
    assert isinstance(deck, Deck)
    assert deck.cards_left == 24
    assert len(set(str(card) for card in deck.cards)) == 24
    allowed = {9, 10, 11, 12, 13, 1}
    assert {card.value.value for card in deck.cards} == allowed
