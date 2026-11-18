import pytest
from gamble.models.cards import Card, Deck
from gamble.errors import InvalidCard


def test_card_get_valid_cards():
    for suit_char in Card.Suits.dict():
        for value_char in Card.Values.dict():
            card_str = value_char + suit_char
            card = Card.get(card_str)
            assert isinstance(card, Card)
            assert card.value.char == value_char
            assert card.suit.char == suit_char


def test_card_get_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("ABC")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("A")


def test_card_get_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("XS")


def test_card_get_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AZ")


def test_card_str_and_repr():
    card = Card.get("AS")
    expected_str = "A♠"
    expected_repr = f"<Card:{expected_str}>"
    assert str(card) == expected_str
    assert repr(card) == expected_repr


def test_card_comparison_operators():
    card1 = Card.get("2S")
    card2 = Card.get("3S")
    card3 = Card.get("2S")
    assert card1 < card2
    assert card2 > card1
    assert card1 <= card3
    assert card1 >= card3
    assert card1 == card3
    assert (card1 == "not a card") is False


def test_card_color_properties():
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    red_suits = [Card.Suits.DIAMONDS, Card.Suits.HEARTS]
    for suit in black_suits:
        card = Card(Card.Values.ACE, suit)
        assert card.color == Card.BLACK
        assert card.is_black
        assert not card.is_red
    for suit in red_suits:
        card = Card(Card.Values.ACE, suit)
        assert card.color == Card.RED
        assert card.is_red
        assert not card.is_black


def test_card_full_name_property():
    card = Card(Card.Values.KING, Card.Suits.DIAMONDS)
    assert card.full_name == "king of diamonds"


def test_card_unicode_property():
    # Test that unicode property returns a string and contains one character
    card_queen = Card(Card.Values.QUEEN, Card.Suits.HEARTS)
    card_king = Card(Card.Values.KING, Card.Suits.CLUBS)
    card_ten = Card(Card.Values.TEN, Card.Suits.SPADES)
    for card in [card_queen, card_king, card_ten]:
        u = card.unicode
        assert isinstance(u, str)
        assert len(u) == 1


def test_deck_default_constructor():
    deck = Deck(shuffle=False)
    assert isinstance(deck.cards, list)
    # 52 cards in a standard deck
    assert len(deck.cards) == 52
    # cards should be reversed order (King spades at bottom if reversed)
    first_card = deck.bottom
    last_card = deck.top
    assert isinstance(first_card, Card)
    assert isinstance(last_card, Card)
    # no shuffles if shuffle is False
    assert deck.shuffles == 0


def test_deck_shuffle_increments_shuffles(monkeypatch):
    deck = Deck(shuffle=False)
    initial_order = deck.cards.copy()

    # Patch random.shuffle to do nothing to check shuffles counter increments
    monkeypatch.setattr("random.shuffle", lambda x: None)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    # Since shuffle does nothing, cards order unchanged
    assert deck.cards == initial_order


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    start_count = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == start_count - 1

    # Draw multiple cards
    cards = deck.draw(times=4)
    assert isinstance(cards, list)
    assert all(isinstance(c, Card) for c in cards)
    assert len(cards) == 4
    assert deck.cards_left == start_count - 5


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    start_count = deck.cards_left
    hand = deck.draw_hand(size=5)
    assert hasattr(hand, "cards")
    assert len(hand.cards) == 5
    assert deck.cards_left == start_count - 5


def test_deck_contains_method():
    deck = Deck(shuffle=False)
    card = deck.cards[0]
    assert card in deck
    not_in_deck = Card(Card.Values.ACE, Card.Suits.SPADES)
    # Remove the card if it appears to guarantee not in deck
    if not_in_deck in deck.cards:
        deck.cards.remove(not_in_deck)
    assert not_in_deck not in deck
    assert "not a card" not in deck


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert isinstance(s, str)
    assert str(deck) == repr(deck)
    assert f"<Deck[{deck.cards_left}]>" == s
