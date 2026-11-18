import pytest
from gamble.models.cards import Card, Deck, EuchreDeck, Suit, Value
from gamble.errors import InvalidCard


def test_card_init_defaults():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_card_get_valid():
    card = Card.get("AS")
    assert isinstance(card, Card)
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


def test_card_get_invalid_length():
    with pytest.raises(InvalidCard):
        Card.get("ASD")
    with pytest.raises(InvalidCard):
        Card.get("A")


def test_card_get_invalid_value():
    with pytest.raises(InvalidCard):
        Card.get("1S")


def test_card_get_invalid_suit():
    with pytest.raises(InvalidCard):
        Card.get("AZ")


def test_card_str_and_repr():
    card = Card.get("KH")
    assert str(card) == "K♥"
    assert repr(card) == "<Card:K♥>"


def test_card_comparisons():
    c1 = Card.get("2C")
    c2 = Card.get("3C")
    c3 = Card.get("2C")

    assert c1 < c2
    assert c2 > c1
    assert c1 <= c3
    assert c2 >= c1
    assert c1 == c3
    assert c1 != c2
    assert (c1 == "not a card") is False


def test_card_color_properties():
    red_suits = [Card.Suits.DIAMONDS, Card.Suits.HEARTS]
    black_suits = [Card.Suits.SPADES, Card.Suits.CLUBS]

    for suit in red_suits:
        card = Card(Card.Values.ACE, suit)
        assert card.color == Card.RED
        assert card.is_red is True
        assert card.is_black is False

    for suit in black_suits:
        card = Card(Card.Values.ACE, suit)
        assert card.color == Card.BLACK
        assert card.is_black is True
        assert card.is_red is False


def test_card_full_name_and_unicode():
    card = Card.get("AH")
    assert card.full_name == "ace of hearts"
    # Unicode should be len 1 and be a string
    uni = card.unicode
    assert isinstance(uni, str)
    assert len(uni) == 1


def test_deck_default_construction():
    deck = Deck(shuffle=False)
    assert len(deck.cards) == 52
    # Cards should be ordered with bottom == 2♠ (lowest) and top == A♥ (highest)
    bottom = deck.bottom
    top = deck.top
    assert bottom.suit == Card.Suits.SPADES
    assert bottom.value == Card.Values.TWO
    assert top.suit == Card.Suits.HEARTS
    assert top.value == Card.Values.ACE
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_deck_shuffle_increments_and_changes_order():
    deck1 = Deck(shuffle=False)
    deck2 = Deck(shuffle=False)
    deck2.shuffle(times=2)
    # Should have shuffles incremented
    assert deck2.shuffles == 2
    # Cards order should differ now mostly
    same_order = all(c1 == c2 for c1, c2 in zip(deck1.cards, deck2.cards))
    assert not same_order


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    num_before = len(deck.cards)

    card = deck.draw()
    assert isinstance(card, Card)
    assert len(deck.cards) == num_before - 1
    assert deck.draws == 1

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert all(isinstance(c, Card) for c in cards)
    assert len(cards) == 3
    assert len(deck.cards) == num_before - 4
    assert deck.draws == 4


def test_deck_contains():
    deck = Deck(shuffle=False)
    card = deck.top
    assert card in deck
    not_in_deck_card = Card(Card.Values.ACE, Card.Suits.SPADES)
    # Remove this card from deck to test absence
    deck.cards.remove(not_in_deck_card)
    assert not_in_deck_card not in deck
    # Non-Card object returns False
    assert "not a card" not in deck


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s == r
    assert s.startswith("<Deck[")
    assert s.endswith("]>")


def test_deck_top_and_bottom_properties():
    deck = Deck(shuffle=False)
    top_card = deck.cards[-1]
    bottom_card = deck.cards[0]
    assert deck.top == top_card
    assert deck.bottom == bottom_card


def test_deck_draw_hand_returns_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Deck.Hand.__bases__[0]) or hasattr(hand, "cards")  # Hand type
    assert len(hand.cards) == 5
    assert all(isinstance(card, Card) for card in hand.cards)


def test_euchre_deck_size_and_cards():
    euchre = EuchreDeck()
    # 6 cards per suit * 4 suits = 24 cards
    assert len(euchre.cards) == 24
    # All cards should be among allowed values
    allowed_values = {1, 9, 10, 11, 12, 13}  # Ace=1, Nine=9, Ten=10, Jack=11, Queen=12, King=13
    for card in euchre.cards:
        assert card.value.value in allowed_values


def test_card_unicode_queen_king_skip_knight():
    # For queens and kings, the unicode calculation hacks (+1)
    queen_card = Card(Card.Values.QUEEN, Card.Suits.SPADES)
    king_card = Card(Card.Values.KING, Card.Suits.SPADES)
    jack_card = Card(Card.Values.JACK, Card.Suits.SPADES)
    ace_card = Card(Card.Values.ACE, Card.Suits.SPADES)

    base_uni = Card.Suits.SPADES.unicode
    # Without hack, unicode for jack: base + 11 (jack value)
    assert ord(str(jack_card.unicode)) == base_uni + 11
    # For queen and king, unicode should be base + value + 1 (hack)
    assert ord(str(queen_card.unicode)) == base_uni + 12 + 1
    assert ord(str(king_card.unicode)) == base_uni + 13 + 1
    # Ace should not apply hack
    assert ord(str(ace_card.unicode)) == base_uni + 1


def test_card_equality_with_non_card():
    card = Card.get("AS")
    assert card != "AS"
    assert card != 1
    assert card != None


def test_card_value_and_suit_enums_all_and_dict():
    suits = Card.Suits.all()
    suits_dict = Card.Suits.dict()
    assert isinstance(suits, list)
    assert all(isinstance(s, Suit) for s in suits)
    assert suits_dict.keys() == {s.char for s in suits}

    values = Card.Values.all()
    values_dict = Card.Values.dict()
    assert isinstance(values, list)
    assert all(isinstance(v, Value) for v in values)
    assert values_dict.keys() == {v.char for v in values}


def test_deck_constructor_with_custom_cards():
    custom_cards = [Card.get("AS"), Card.get("KH")]
    deck = Deck(cards=custom_cards, shuffle=False)
    assert len(deck.cards) == 2
    assert deck.cards[-1] == Card.get("KH")
    assert deck.cards[0] == Card.get("AS")
    assert deck.shuffles == 0
