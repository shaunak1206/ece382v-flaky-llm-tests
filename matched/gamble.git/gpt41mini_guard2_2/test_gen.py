import random
import pytest
from gamble.models.cards import Card, Deck, Hand
from gamble.errors import InvalidCard


def test_card_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    values = [s.value for s in suits]
    assert values == sorted(values)
    # Check that all 4 suits are present and unique
    assert len(suits) == 4
    assert len(set(suits)) == 4
    # Check properties of known suits
    suit_names = {s.name for s in suits}
    expected_names = {"spades", "clubs", "diamonds", "hearts"}
    assert suit_names == expected_names


def test_card_values_all_sorted_by_value():
    values = Card.Values.all()
    vals = [v.value for v in values]
    assert vals == sorted(vals)
    # Check that all 13 values are present
    assert len(values) == 13
    # Check ACE is 1 and KING is 13
    assert values[0].char == "A" and values[0].value == 1
    assert values[-1].char == "K" and values[-1].value == 13


@pytest.mark.parametrize("text, expected_value_char, expected_suit_char", [
    ("AS", "A", "S"),
    ("TD", "T", "D"),
    ("7H", "7", "H"),
    ("Qc", "Q", "C"),
])
def test_card_get_valid_cards(text, expected_value_char, expected_suit_char):
    card = Card.get(text)
    assert card.value.char == expected_value_char
    assert card.suit.char == expected_suit_char.upper()
    # string representation matches value char + suit symbol
    assert str(card).startswith(expected_value_char)
    assert card.suit.char == expected_suit_char.upper()


@pytest.mark.parametrize("text", [
    "", "A", "ASD", "123", "ZZ",
    "1S", "AB", "XC", "A1", "AA", "SS"
])
def test_card_get_invalid_cards_raises(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_color_properties():
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    red_suits = [Card.Suits.DIAMONDS, Card.Suits.HEARTS]
    for suit in black_suits:
        card = Card(value=Card.Values.ACE, suit=suit)
        assert card.color == Card.BLACK
        assert card.is_black is True
        assert card.is_red is False
    for suit in red_suits:
        card = Card(value=Card.Values.ACE, suit=suit)
        assert card.color == Card.RED
        assert card.is_red is True
        assert card.is_black is False


@pytest.mark.parametrize("value_char", [v.char for v in Card.Values.all()])
@pytest.mark.parametrize("suit", Card.Suits.all())
def test_card_unicode_property(value_char, suit):
    card = Card(value=Card.Values.dict()[value_char], suit=suit)
    codepoint = ord(card.unicode)
    base = suit.unicode
    val = card.value.value
    # For values >= 12 (queen=12, king=13), unicode offset is +1
    expected_offset = val + (1 if val >= 12 else 0)
    assert codepoint == base + expected_offset


def test_card_comparisons():
    ace = Card(Card.Values.ACE, Card.Suits.HEARTS)
    king = Card(Card.Values.KING, Card.Suits.CLUBS)
    ace2 = Card(Card.Values.ACE, Card.Suits.SPADES)
    # lt and gt based on value only, suits don't matter
    assert ace < king
    assert king > ace
    assert ace <= ace2
    assert ace >= ace2
    # equality considers value and suit
    assert ace == ace
    assert ace != king
    assert ace != ace2  # different suits


def test_deck_default_creation_and_card_counts(monkeypatch):
    # Seed random to ensure shuffle predictable but don't care about order here
    random.seed(42)
    deck = Deck(shuffle=False)
    assert len(deck.cards) == 52
    # Cards should be reversed default (king high at top)
    top_card = deck.top
    bottom_card = deck.bottom
    # Top card should be highest value, lowest suit (SPADES=0): ACE of Spades reversed means King of Spades at top
    assert top_card.value.value == 1  # ACE has value 1 but card list reversed
    assert bottom_card.value.value == 13  # KING
    # Because of reverse, last is ACE of SPADES, first is KING of HEARTS
    assert bottom_card.suit.name == "hearts"
    assert top_card.suit.name == "spades"


def test_deck_contains_cards(monkeypatch):
    deck = Deck(shuffle=False)
    card = Card.get("AS")
    assert card in deck
    non_card = "not_a_card"
    assert non_card not in deck
    different_card = Card(Card.Values.TWO, Card.Suits.HEARTS)
    assert different_card in deck
    # Remove card and check it's no longer in deck
    deck.cards.remove(card)
    assert card not in deck


def test_deck_draw_single_and_multiple(monkeypatch):
    random.seed(123)
    deck = Deck(shuffle=False)
    original_count = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == original_count - 1
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == original_count - 4


def test_deck_draw_hand(monkeypatch):
    random.seed(456)
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert all(isinstance(card, Card) for card in hand.cards)
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_times_increments_shuffles(monkeypatch):
    seed = 789
    random.seed(seed)
    deck = Deck(shuffle=False)
    initial_shuffles = deck.shuffles
    deck.shuffle(times=3)
    assert deck.shuffles == initial_shuffles + 3


def test_hand_rank_properties():
    # Test one pair
    hand = Hand.get("AS, AH, 2D, 3C, 4S")
    assert hand.is_one_pair
    assert not hand.is_two_pair
    # Test two pair
    hand = Hand.get("AS, AH, 2D, 2C, 4S")
    assert hand.is_two_pair
    # Test three of a kind
    hand = Hand.get("AS, AH, AD, 2C, 4S")
    assert hand.is_three_of_a_kind
    # Test four of a kind
    hand = Hand.get("AS, AH, AD, AC, 4S")
    assert hand.is_four_of_a_kind
    # Test full house
    hand = Hand.get("AS, AH, AD, 4C, 4S")
    assert hand.is_full_house
    # Test flush
    hand = Hand.get("AS, 3S, 5S, 7S, 9S")
    assert hand.is_flush
    # Test straight
    hand = Hand.get("9S, TS, JD, QC, KH")
    assert hand.is_straight
    # Test straight flush
    hand = Hand.get("9S, TS, JS, QS, KS")
    assert hand.is_straight_flush
    # Test high card (no combination)
    hand = Hand.get("AS, 3H, 5D, 7C, 9S")
    assert hand.rank.name == "high card"


def test_hand_len_and_str_and_repr():
    hand = Hand.get("AS, 2H, 3D, 4C, 5S")
    assert len(hand) == 5
    s = str(hand)
    assert "[AS, 2H, 3D, 4C, 5S]" == s
    r = repr(hand)
    assert r.startswith("<Hand[5](")
    assert str(hand) in r


def test_deck_draw_empties_deck_and_raises(monkeypatch):
    random.seed(1001)
    deck = Deck(shuffle=False)
    count = deck.cards_left
    for _ in range(count):
        deck.draw()
    assert deck.cards_left == 0
    # Drawing from empty deck will raise IndexError
    with pytest.raises(IndexError):
        deck.draw()


def test_card_repr_and_str_consistency():
    card = Card.get("QS")
    s = str(card)
    r = repr(card)
    assert r == "<Card:{}>".format(s)
    assert len(s) == 2
    assert s[1] == card.suit.symbol


def test_card_get_case_insensitivity():
    card1 = Card.get("as")
    card2 = Card.get("AS")
    card3 = Card.get("aS")
    card4 = Card.get("As")
    # All forms create same card (value and suit)
    for c in (card1, card2, card3, card4):
        assert isinstance(c, Card)
    assert all(c == card1 for c in (card2, card3, card4))


def test_deck_default_order_unshuffled(monkeypatch):
    deck = Deck(shuffle=False)
    cards = deck.cards[:]
    # Cards should be all 52 unique
    unique_cards = set(str(c) for c in cards)
    assert len(unique_cards) == 52
    # Cards sorted reversed by default (last card in orig order is first in deck)
    sorted_cards = sorted(cards, key=lambda c: (c.suit.value, c.value.value))
    assert cards == sorted_cards[::-1]


def test_deck_str_and_repr_contains_cards_left():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert str(deck.cards_left) in s
    assert s == r


def test_deck_shuffle_randomness(monkeypatch):
    # Seed random and check that shuffle really shuffles
    random.seed(12345)
    deck1 = Deck(shuffle=False)
    cards_before = deck1.cards[:]
    deck1.shuffle()
    cards_after = deck1.cards
    # After shuffle, order should be different in almost all runs
    # We allow the (rare) possibility that order is same, but test that cards are same set
    assert set(cards_before) == set(cards_after)
    # It's very unlikely that order is identical after shuffle, test that
    assert cards_before != cards_after


def test_card_full_name_property():
    card = Card.get("KH")
    full_name = card.full_name
    expected = "king of hearts"
    assert full_name == expected


def test_card_eq_with_non_card():
    card = Card.get("KS")
    assert card != "KS"
    assert card != 10
    assert card != None
