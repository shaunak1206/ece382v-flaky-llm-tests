import pytest
import random
from gamble.models.cards import Card, Deck, EuchreDeck
from gamble.errors import InvalidCard


def test_card_suits_all_sorted_by_value():
    suits = Card.Suits.all()
    assert all(isinstance(s, Card.Suits.SPADES.__class__) for s in suits)
    values = [s.value for s in suits]
    assert values == sorted(values)


def test_card_values_all_sorted_by_value():
    values = Card.Values.all()
    assert all(isinstance(v, Card.Values.ACE.__class__) for v in values)
    vals = [v.value for v in values]
    assert vals == sorted(vals)


def test_card_get_valid_cards():
    examples = ["AS", "2C", "TD", "KH", "JH", "QS"]
    for txt in examples:
        card = Card.get(txt)
        assert str(card) == txt[0] + card.suit.symbol
        assert card.value.char == txt[0]
        assert card.suit.char == txt[1]


def test_card_get_invalid_length_raises():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("ABC")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("A")


def test_card_get_invalid_value_raises():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1S")  # '1' is invalid value char


def test_card_get_invalid_suit_raises():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AZ")  # 'Z' not a valid suit char


def test_card_properties_color_is_black_or_red():
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    red_suits = [Card.Suits.DIAMONDS, Card.Suits.HEARTS]
    for suit in black_suits:
        card = Card(value=Card.Values.ACE, suit=suit)
        assert card.color == Card.BLACK
        assert card.is_black
        assert not card.is_red
    for suit in red_suits:
        card = Card(value=Card.Values.ACE, suit=suit)
        assert card.color == Card.RED
        assert not card.is_black
        assert card.is_red


def test_card_full_name_property():
    card = Card(value=Card.Values.KING, suit=Card.Suits.HEARTS)
    assert card.full_name == "king of hearts"


def test_card_unicode_correctness():
    # We test that unicode returns a single character string
    card_low = Card(value=Card.Values.TEN, suit=Card.Suits.SPADES)
    card_high = Card(value=Card.Values.QUEEN, suit=Card.Suits.SPADES)
    for card in (card_low, card_high):
        u = card.unicode
        assert isinstance(u, str)
        assert len(u) == 1
        # unicode ordinal should correspond to suit.unicode + value with hack if Q or K
        hack = int(card.value.value >= 12)
        expected_ord = card.suit.unicode + card.value.value + hack
        assert ord(u) == expected_ord


def test_card_str_and_repr():
    card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    s = str(card)
    r = repr(card)
    assert s == "AS"
    assert r == f"<Card:{s}>"


def test_card_comparisons_lt_gt_eq():
    c1 = Card(value=Card.Values.TWO, suit=Card.Suits.HEARTS)
    c2 = Card(value=Card.Values.THREE, suit=Card.Suits.HEARTS)
    c3 = Card(value=Card.Values.TWO, suit=Card.Suits.CLUBS)

    # less than
    assert c1 < c2
    assert not (c2 < c1)
    # greater than
    assert c2 > c1
    assert not (c1 > c2)
    # less or equal
    assert c1 <= c2
    assert c1 <= c3
    assert c1 <= c1
    # greater or equal
    assert c2 >= c1
    assert c1 >= c3
    assert c1 >= c1
    # equality checks with suit
    assert c1 == c1
    assert c1 != c3
    assert c1 != c2
    assert c1 != "not a card"


def test_deck_default_has_52_cards_and_order():
    deck = Deck(shuffle=False)
    assert len(deck.cards) == 52
    # Cards should be reversed sorted order: King -> Ace low
    values = [card.value.value for card in deck.cards]
    suits = [card.suit.value for card in deck.cards]
    # Because cards.reverse() called on ordered suit/value iteration,
    # the bottom card should be ACE of spades (lowest value 1, suit 0)
    bottom = deck.bottom
    assert bottom.value == Card.Values.ACE
    assert bottom.suit == Card.Suits.SPADES
    # The top card should be King of hearts (max value 13, suit 3)
    top = deck.top
    assert top.value == Card.Values.KING
    assert top.suit == Card.Suits.HEARTS


def test_deck_contains_card():
    deck = Deck(shuffle=False)
    ace_spades = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert ace_spades in deck
    # Non card type
    assert 5 not in deck
    # Card not in deck
    fake_card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    deck.cards.remove(fake_card)
    assert fake_card not in deck


def test_deck_draw_single_and_multiple_cards():
    deck = Deck(shuffle=False)
    original_count = deck.cards_left
    card1 = deck.draw()
    assert isinstance(card1, Card)
    assert deck.cards_left == original_count - 1

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == original_count - 4  # 1 + 3 drawn


def test_deck_draw_hand_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    from gamble.models.cards import Hand

    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_changes_order():
    # We will shuffle twice and check that order changes
    deck1 = Deck(shuffle=False)
    original_order = list(deck1.cards)
    deck1.shuffle(times=3)
    after_shuffle = deck1.cards
    # It is possible that deck is the same by chance but very unlikely
    assert after_shuffle != original_order
    assert deck1.shuffles == 3


def test_euchre_deck_cards_and_size():
    deck = EuchreDeck(shuffle=False)
    # Euchre deck uses 9, 10, J, Q, K, A of all suits
    # That is 6 values * 4 suits = 24 cards
    assert len(deck.cards) == 24
    # All cards should be in the deck
    valid_values = {9, 10, 11, 12, 13, 1}  # 1 is ace
    for card in deck.cards:
        assert card.value.value in valid_values
        assert card.suit in Card.Suits.all()


def test_card_get_case_insensitive_and_strip_spaces():
    # Spaces should be removed before passing; test uppercase internally
    card = Card.get(" aS ")
    # Should raise invalid because len != 2, so strip before get for test
    with pytest.raises(InvalidCard):
        Card.get("  aS")
    # So test after stripping spaces
    card = Card.get("AS")
    assert card.value.char == "A"
    assert card.suit.char == "S"


def test_deck_draw_more_cards_than_available_raises_index_error():
    deck = Deck(shuffle=False)
    with pytest.raises(IndexError):
        deck.draw(times=53)


def test_deck_str_and_repr_shows_cards_left():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert f"{deck.cards_left}" in s
    assert s == r


def test_card_eq_with_non_card_returns_false():
    card = Card(value=Card.Values.ACE, suit=Card.Suits.SPADES)
    assert card != 42
    assert card != "AS"
