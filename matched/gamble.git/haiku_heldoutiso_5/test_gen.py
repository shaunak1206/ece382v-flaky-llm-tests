import pytest

from gamble.errors import InvalidCard
from gamble.models import cards as cards_module
from gamble.models.cards import Card, Deck, EuchreDeck


# ---------------------------------------------------------------------------
# Card: construction and text parsing
# ---------------------------------------------------------------------------

def test_default_card_is_ace_of_spades():
    card = Card()
    assert card.value is Card.Values.ACE
    assert card.suit is Card.Suits.SPADES


def test_get_parses_value_and_suit():
    card = Card.get("KH")
    assert card.value is Card.Values.KING
    assert card.suit is Card.Suits.HEARTS


def test_get_is_case_insensitive():
    assert Card.get("qd") == Card(value=Card.Values.QUEEN, suit=Card.Suits.DIAMONDS)


def test_get_ten_uses_T_character():
    card = Card.get("TC")
    assert card.value is Card.Values.TEN
    assert card.suit is Card.Suits.CLUBS


@pytest.mark.parametrize("text", ["", "A", "AS1", "10SS", "ASDF"])
def test_get_rejects_wrong_length(text):
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get(text)


def test_get_rejects_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("XS")


def test_get_rejects_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AX")


# ---------------------------------------------------------------------------
# Card: properties and representations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "suit, expected_color, is_black, is_red",
    [
        (Card.Suits.SPADES, Card.BLACK, True, False),
        (Card.Suits.CLUBS, Card.BLACK, True, False),
        (Card.Suits.DIAMONDS, Card.RED, False, True),
        (Card.Suits.HEARTS, Card.RED, False, True),
    ],
)
def test_color_and_color_predicates_follow_suit(suit, expected_color, is_black, is_red):
    card = Card(value=Card.Values.FIVE, suit=suit)
    assert card.color == expected_color
    assert card.is_black is is_black
    assert card.is_red is is_red


def test_full_name():
    assert Card(value=Card.Values.ACE, suit=Card.Suits.SPADES).full_name == "ace of spades"


def test_str_uses_value_char_and_suit_symbol():
    assert str(Card.get("TH")) == "T♥"


def test_repr_wraps_str():
    assert repr(Card.get("AS")) == "<Card:A♠>"


def test_unicode_for_ace_of_spades():
    assert Card.get("AS").unicode == "\U0001F0A1"


def test_unicode_skips_knight_for_queen_and_king():
    # Queen of spades is U+1F0AD and king of spades is U+1F0AE in the playing
    # card block, so the knight (U+1F0AC) is skipped for face cards.
    assert Card.get("QS").unicode == "\U0001F0AD"
    assert Card.get("KS").unicode == "\U0001F0AE"


# ---------------------------------------------------------------------------
# Card: comparison and equality
# ---------------------------------------------------------------------------

def test_ordering_uses_value_ranking():
    ace = Card(value=Card.Values.ACE)
    two = Card(value=Card.Values.TWO)
    king = Card(value=Card.Values.KING)
    assert ace < two
    assert two < king
    assert king > ace
    assert not (king < ace)


def test_less_than_or_equal_and_greater_than_or_equal_are_inclusive():
    seven = Card(value=Card.Values.SEVEN, suit=Card.Suits.SPADES)
    same = Card(value=Card.Values.SEVEN, suit=Card.Suits.SPADES)
    nine = Card(value=Card.Values.NINE, suit=Card.Suits.SPADES)
    assert seven <= same
    assert seven >= same
    assert seven <= nine
    assert nine >= seven
    assert not (nine <= seven)
    assert not (seven >= nine)


def test_equality_requires_same_value_and_suit():
    a = Card(value=Card.Values.JACK, suit=Card.Suits.HEARTS)
    b = Card(value=Card.Values.JACK, suit=Card.Suits.HEARTS)
    assert a == b


def test_inequality_when_suit_differs():
    a = Card(value=Card.Values.JACK, suit=Card.Suits.HEARTS)
    b = Card(value=Card.Values.JACK, suit=Card.Suits.SPADES)
    assert a != b


def test_equality_with_non_card_is_false():
    assert Card() != "AS"
    assert (Card() == 1) is False


# ---------------------------------------------------------------------------
# Enums: Suits and Values
# ---------------------------------------------------------------------------

def test_suits_all_returns_four_suits_sorted_by_value():
    suits = Card.Suits.all()
    assert [s.name for s in suits] == ["spades", "clubs", "diamonds", "hearts"]


def test_suits_dict_maps_char_to_suit():
    mapping = Card.Suits.dict()
    assert mapping["S"] is Card.Suits.SPADES
    assert mapping["H"] is Card.Suits.HEARTS
    assert len(mapping) == 4


def test_values_all_returns_thirteen_values_sorted():
    values = Card.Values.all()
    assert len(values) == 13
    assert [v.value for v in values] == list(range(1, 14))
    assert values[0] is Card.Values.ACE
    assert values[-1] is Card.Values.KING


def test_values_dict_maps_char_to_value():
    mapping = Card.Values.dict()
    assert mapping["T"] is Card.Values.TEN
    assert mapping["A"] is Card.Values.ACE
    assert len(mapping) == 13


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

def test_default_deck_has_52_unique_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    assert len({str(card) for card in deck.cards}) == 52


def test_unshuffled_deck_top_and_bottom_cards():
    deck = Deck(shuffle=False)
    assert deck.top == Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert deck.bottom == Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)


def test_shuffle_flag_controls_shuffle_count():
    assert Deck(shuffle=False).shuffles == 0
    assert Deck(shuffle=True).shuffles == 1


def test_shuffle_calls_random_shuffle_the_given_number_of_times(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cards_module.random, "shuffle", lambda seq: calls.append(len(seq))
    )
    deck = Deck(cards=[Card(), Card.get("KH")], shuffle=False)
    deck.shuffle(times=3)
    assert calls == [2, 2, 2]
    assert deck.shuffles == 3


def test_shuffle_preserves_the_same_cards():
    deck = Deck(shuffle=False)
    before = sorted(str(c) for c in deck.cards)
    deck.shuffle()
    after = sorted(str(c) for c in deck.cards)
    assert before == after


def test_draw_single_card_returns_top_card_and_updates_counters():
    deck = Deck(shuffle=False)
    top = deck.top
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert drawn == top
    assert deck.cards_left == 51
    assert deck.draws == 1


def test_draw_multiple_cards_returns_list_in_draw_order():
    deck = Deck(shuffle=False)
    expected = list(reversed(deck.cards[-3:]))
    drawn = deck.draw(times=3)
    assert isinstance(drawn, list)
    assert drawn == expected
    assert deck.cards_left == 49
    assert deck.draws == 3


def test_draw_from_empty_deck_raises_index_error():
    deck = Deck(cards=[Card()], shuffle=False)
    deck.draw()
    with pytest.raises(IndexError):
        deck.draw()


def test_draw_hand_returns_hand_of_requested_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert len(hand) == 5
    assert hand.size == 5
    assert deck.cards_left == 47
    assert deck.draws == 5


def test_draw_hand_of_one_card_still_returns_hand():
    deck = Deck(cards=[Card.get("2C"), Card.get("3D")], shuffle=False)
    hand = deck.draw_hand(size=1)
    assert len(hand) == 1
    assert hand.cards[0] == Card.get("3D")


def test_contains_detects_cards_in_deck_and_rejects_non_cards():
    deck = Deck(cards=[Card.get("AS"), Card.get("KH")], shuffle=False)
    assert Card.get("KH") in deck
    assert Card.get("2C") not in deck
    assert "AS" not in deck


def test_str_and_repr_show_cards_left():
    deck = Deck(shuffle=False)
    deck.draw(times=2)
    assert str(deck) == "<Deck[50]>"
    assert repr(deck) == "<Deck[50]>"


def test_euchre_deck_contains_24_cards_of_nine_through_ace():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    allowed_values = {9, 10, 11, 12, 13, 1}
    assert all(card.value.value in allowed_values for card in deck.cards)
    assert len({str(card) for card in deck.cards}) == 24
    assert deck.shuffles == 1
