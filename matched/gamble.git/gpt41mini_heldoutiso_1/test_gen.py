import pytest
from gamble.models.cards import Card, Deck, EuchreDeck, Card as card_module
from gamble.errors import InvalidCard


def test_card_init_defaults():
    c = Card()
    assert c.value == Card.Values.ACE
    assert c.suit == Card.Suits.SPADES


def test_card_get_valid_cards():
    for val in Card.Values.all():
        for suit in Card.Suits.all():
            text = val.char + suit.char
            c = Card.get(text)
            assert c.value == val
            assert c.suit == suit


def test_card_get_invalid_length():
    with pytest.raises(InvalidCard):
        Card.get("A")
    with pytest.raises(InvalidCard):
        Card.get("ASD")
    with pytest.raises(InvalidCard):
        Card.get("")


def test_card_get_invalid_value():
    with pytest.raises(InvalidCard):
        Card.get("1S")  # '1' not a valid card char


def test_card_get_invalid_suit():
    with pytest.raises(InvalidCard):
        Card.get("AZ")  # 'Z' not a valid suit


def test_card_color_properties():
    red_suits = [Card.Suits.DIAMONDS, Card.Suits.HEARTS]
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]
    for suit in red_suits:
        c = Card(Card.Values.ACE, suit)
        assert c.color == Card.RED
        assert c.is_red is True
        assert c.is_black is False
    for suit in black_suits:
        c = Card(Card.Values.ACE, suit)
        assert c.color == Card.BLACK
        assert c.is_black is True
        assert c.is_red is False


def test_card_full_name_property():
    c = Card(Card.Values.KING, Card.Suits.HEARTS)
    assert c.full_name == "king of hearts"
    c = Card(Card.Values.TWO, Card.Suits.SPADES)
    assert c.full_name == "two of spades"


def test_card_unicode_property():
    # Test normal value under 12 (queen), no hack
    c = Card(Card.Values.TEN, Card.Suits.CLUBS)
    base_unicode = Card.Suits.CLUBS.unicode + Card.Values.TEN.value
    assert ord(c.unicode) == base_unicode
    # Test value 12 or above (queen and king), hack applied
    q = Card(Card.Values.QUEEN, Card.Suits.CLUBS)
    k = Card(Card.Values.KING, Card.Suits.CLUBS)
    assert ord(q.unicode) == Card.Suits.CLUBS.unicode + 12 + 1
    assert ord(k.unicode) == Card.Suits.CLUBS.unicode + 13 + 1


def test_card_str_and_repr():
    c = Card(Card.Values.JACK, Card.Suits.DIAMONDS)
    s = str(c)
    r = repr(c)
    assert s == "J♦"
    assert r == f"<Card:{s}>"


def test_card_comparisons():
    c1 = Card(Card.Values.TWO, Card.Suits.CLUBS)
    c2 = Card(Card.Values.THREE, Card.Suits.CLUBS)
    c3 = Card(Card.Values.TWO, Card.Suits.HEARTS)
    assert c1 < c2
    assert c2 > c1
    assert c1 <= c3  # equal value
    assert c1 >= c3  # equal value
    assert c1 == Card(Card.Values.TWO, Card.Suits.CLUBS)
    assert c1 != c2
    assert c1 != "not a card"


def test_deck_default_constructor_52_cards():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    # The deck.cards should contain unique cards and be standard deck
    suits = set(c.suit for c in deck.cards)
    values = set(c.value for c in deck.cards)
    assert suits == set(Card.Suits.all())
    assert values == set(Card.Values.all())


def test_deck_shuffle_and_shuffles_counter():
    deck = Deck(shuffle=False)
    original_order = deck.cards.copy()
    deck.shuffle(times=2)
    assert deck.shuffles == 2
    # After shuffle cards order should likely differ
    assert deck.cards != original_order


def test_deck_draw_single_and_multiple_cards():
    deck = Deck(shuffle=False)
    init_len = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == init_len - 1
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert all(isinstance(c, Card) for c in cards)
    assert deck.cards_left == init_len - 4
    assert deck.draws == 4


def test_deck_draw_hand_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert hasattr(hand, "cards")
    assert len(hand.cards) == 5
    assert isinstance(hand.cards[0], Card)
    assert deck.cards_left == 52 - 5


def test_deck_contains_operator():
    deck = Deck(shuffle=False)
    card = Card(Card.Values.ACE, Card.Suits.SPADES)
    assert card in deck
    not_a_card = "AS"
    assert not_a_card not in deck
    fake_card = Card(Card.Values.ACE, Card.Suits.SPADES)
    deck.cards.remove(fake_card)
    assert fake_card not in deck


def test_deck_top_and_bottom_properties():
    deck = Deck(shuffle=False)
    top_card = deck.top
    bottom_card = deck.bottom
    assert top_card == deck.cards[-1]
    assert bottom_card == deck.cards[0]


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s == r
    assert s.startswith("<Deck[")
    assert s.endswith("]>")


def test_euchre_deck_contains_only_9_to_ace_and_24_cards():
    deck = EuchreDeck()
    values = set(c.value.value for c in deck.cards)
    suits = set(c.suit for c in deck.cards)
    # Euchre Deck includes 9,10,J,Q,K,A (values 9,10,11,12,13,1)
    allowed_values = set([1, 9, 10, 11, 12, 13])
    assert values.issubset(allowed_values)
    assert len(deck.cards) == 24
    assert suits == set(Card.Suits.all())
    # top/bottom also available
    assert isinstance(deck.top, Card)
    assert isinstance(deck.bottom, Card)
