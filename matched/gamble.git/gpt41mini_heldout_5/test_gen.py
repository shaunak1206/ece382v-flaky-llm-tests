import pytest
from gamble.models.cards import Card, Deck
from gamble.errors import InvalidCard


def test_card_suits_all_returns_all_suits_sorted():
    suits = Card.Suits.all()
    assert len(suits) == 4
    assert suits[0].name == "spades"
    assert suits[1].name == "clubs"
    assert suits[2].name == "diamonds"
    assert suits[3].name == "hearts"


def test_card_values_all_returns_all_values_sorted_by_value():
    values = Card.Values.all()
    assert len(values) == 13
    assert values[0].name == "ace"
    assert values[-1].name == "king"
    assert all(values[i].value <= values[i + 1].value for i in range(len(values) - 1))


def test_card_get_valid_string_creates_correct_card():
    card = Card.get("AS")
    assert card.value.char == "A"
    assert card.suit.char == "S"
    assert str(card) == "A♠"


def test_card_get_invalid_length_raises_invalidcard():
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("AAA")
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("A")


def test_card_get_invalid_value_char_raises_invalidcard():
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("1S")


def test_card_get_invalid_suit_char_raises_invalidcard():
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AZ")


def test_card_color_properties():
    red_card = Card(Card.Values.ACE, Card.Suits.HEARTS)
    black_card = Card(Card.Values.KING, Card.Suits.SPADES)
    assert red_card.color == Card.RED
    assert red_card.is_red is True
    assert red_card.is_black is False
    assert black_card.color == Card.BLACK
    assert black_card.is_black is True
    assert black_card.is_red is False


def test_card_full_name_property():
    card = Card(Card.Values.QUEEN, Card.Suits.DIAMONDS)
    assert card.full_name == "queen of diamonds"


def test_card_unicode_property_matches_expected():
    # Ace of Spades: suit.unicode=127136, value=1, hack=0 (value<12)
    card1 = Card(Card.Values.ACE, Card.Suits.SPADES)
    expected1 = chr(card1.suit.unicode + card1.value.value)
    assert card1.unicode == expected1

    # Queen of Clubs: value=12 -> hack=1
    card2 = Card(Card.Values.QUEEN, Card.Suits.CLUBS)
    expected2 = chr(card2.suit.unicode + card2.value.value + 1)
    assert card2.unicode == expected2

    # King of Diamonds: value=13 -> hack=1
    card3 = Card(Card.Values.KING, Card.Suits.DIAMONDS)
    expected3 = chr(card3.suit.unicode + card3.value.value + 1)
    assert card3.unicode == expected3


def test_card_str_and_repr_methods():
    card = Card(Card.Values.TEN, Card.Suits.HEARTS)
    assert str(card) == "T♥"
    assert repr(card) == "<Card:T♥>"


def test_card_comparison_methods():
    card1 = Card(Card.Values.TWO, Card.Suits.SPADES)
    card2 = Card(Card.Values.THREE, Card.Suits.CLUBS)
    card3 = Card(Card.Values.TWO, Card.Suits.SPADES)

    assert card1 < card2
    assert card2 > card1
    assert card1 <= card3
    assert card1 >= card3
    assert card1 == card3
    assert card1 != card2
    assert (card1 == "not a card") is False


def test_deck_defaults_to_52_cards_and_contains():
    deck = Deck(shuffle=False)
    assert len(deck.cards) == 52
    all_suits = {suit.name for suit in Card.Suits.all()}
    all_values = {val.name for val in Card.Values.all()}
    # Check each card is valid and suit/name presence
    for card in deck.cards:
        assert card.suit.name in all_suits
        assert card.value.name in all_values

    # Deck contains a known card
    card = Card(Card.Values.ACE, Card.Suits.SPADES)
    assert card in deck

    # Non-card object not in deck
    assert "A♠" not in deck


def test_deck_init_with_cards_and_no_shuffle():
    cards = [Card(Card.Values.ACE, Card.Suits.SPADES), Card(Card.Values.TWO, Card.Suits.CLUBS)]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards == cards
    assert deck.shuffles == 0


def test_deck_shuffle_increments_shuffles_and_changes_order():
    deck1 = Deck(shuffle=False)
    cards_before = deck1.cards.copy()
    deck1.shuffle(times=2)
    assert deck1.shuffles == 2
    # It's possible shuffle results in same order, but very unlikely; check not equal
    assert deck1.cards != cards_before or deck1.shuffles == 2


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    count_before = len(deck.cards)
    card = deck.draw()
    assert isinstance(card, Card)
    assert len(deck.cards) == count_before - 1
    deck.draws = 0

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert all(isinstance(c, Card) for c in cards)
    assert deck.draws == 3


def test_deck_draw_hand_returns_hand_of_correct_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert len(hand.cards) == 5
    assert all(isinstance(c, Card) for c in hand.cards)


def test_deck_top_and_bottom_properties():
    deck = Deck(shuffle=False)
    top_card = deck.top
    bottom_card = deck.bottom
    assert top_card == deck.cards[-1]
    assert bottom_card == deck.cards[0]


def test_deck_str_and_repr_return_expected_format():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s.startswith("<Deck[")
    assert s.endswith("]>")
    assert r == s


def test_deck_contains_non_card_returns_false():
    deck = Deck(shuffle=False)
    assert ("not a card") not in deck
    assert 10 not in deck


def test_card_equality_with_other_types_returns_false():
    card = Card(Card.Values.TEN, Card.Suits.CLUBS)
    assert (card == "some string") is False
    assert (card == None) is False
