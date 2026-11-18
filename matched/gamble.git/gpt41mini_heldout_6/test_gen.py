import pytest
from gamble.models.cards import Card, Deck, EuchreDeck
from gamble.errors import InvalidCard


def test_card_get_valid():
    card = Card.get("AS")
    assert card.value.char == "A"
    assert card.suit.char == "S"
    assert str(card) == "A♠"
    assert repr(card).startswith("<Card:")


def test_card_get_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("ASD")
    with pytest.raises(InvalidCard, match="Too many characters"):
        Card.get("A")


def test_card_get_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value"):
        Card.get("1S")


def test_card_get_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit"):
        Card.get("AZ")


def test_card_color_properties():
    card_black = Card(Card.Values.ACE, Card.Suits.CLUBS)
    card_red = Card(Card.Values.KING, Card.Suits.HEARTS)
    assert card_black.color == Card.BLACK
    assert card_black.is_black is True
    assert card_black.is_red is False
    assert card_red.color == Card.RED
    assert card_red.is_red is True
    assert card_red.is_black is False


def test_card_unicode_property_matches_expected():
    # Spot check for ACE and KING of SPADES (black suit)
    ace_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    king_spades = Card(Card.Values.KING, Card.Suits.SPADES)
    # unicode should be a single char string
    assert isinstance(ace_spades.unicode, str)
    assert len(ace_spades.unicode) == 1
    assert isinstance(king_spades.unicode, str)
    assert len(king_spades.unicode) == 1
    # Unicode chars for king should be higher or equal than ace
    assert ord(king_spades.unicode) >= ord(ace_spades.unicode)


def test_card_comparisons():
    card_ace_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    card_king_spades = Card(Card.Values.KING, Card.Suits.SPADES)
    card_ace_hearts = Card(Card.Values.ACE, Card.Suits.HEARTS)

    assert card_ace_spades < card_king_spades
    assert card_king_spades > card_ace_spades
    assert card_ace_spades <= card_ace_spades
    assert card_king_spades >= card_king_spades
    assert card_ace_spades == card_ace_spades
    assert card_ace_spades != card_ace_hearts
    assert (card_ace_spades == "not a card") is False


def test_deck_init_default_deck():
    deck = Deck(shuffle=False)
    assert deck.cards_left == 52
    # top card value should be the highest value (KING) of HEARTS (last suit)
    top = deck.top
    assert top.suit == Card.Suits.HEARTS
    assert top.value == Card.Values.KING
    # bottom card is ACE of SPADE (first suit)
    bottom = deck.bottom
    assert bottom.suit == Card.Suits.SPADES
    assert bottom.value == Card.Values.ACE
    assert deck.shuffles == 0
    assert deck.draws == 0


def test_deck_init_with_cards_and_no_shuffle():
    ace_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    king_hearts = Card(Card.Values.KING, Card.Suits.HEARTS)
    deck = Deck(cards=[ace_spades, king_hearts], shuffle=False)
    assert deck.cards_left == 2
    assert deck.top == king_hearts
    assert deck.bottom == ace_spades
    assert deck.shuffles == 0


def test_deck_shuffle_increases_shuffles_and_changes_order():
    deck = Deck(shuffle=False)
    orig_order = deck.cards[:]
    deck.shuffle(times=2)
    assert deck.shuffles == 2
    # The order should most likely change; since shuffle is random, this could fail rarely but is acceptable
    assert deck.cards != orig_order


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    num_before = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == num_before - 1

    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == num_before - 4


def test_deck_contains_method():
    deck = Deck(shuffle=False)
    ace_spades = Card(Card.Values.ACE, Card.Suits.SPADES)
    two_hearts = Card(Card.Values.TWO, Card.Suits.HEARTS)
    assert ace_spades in deck
    assert two_hearts in deck
    other_card = Card(Card.Values.ACE, Card.Suits.HEARTS)
    assert other_card in deck
    not_in_deck = Card(Card.Values.ACE, Card.Suits.HEARTS)
    deck.cards.remove(not_in_deck)
    assert not_in_deck not in deck
    # Non-card is always not in deck
    assert "AS" not in deck


def test_deck_draw_hand_returns_hand_and_correct_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    from gamble.models.cards import Hand  # this is safe since Hand is imported in cards.py
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 52 - 5


def test_euchre_deck_contains_only_euchre_cards():
    deck = EuchreDeck()
    valid_values = {x.char for x in Card.Values.all() if x.value >= 9 or x.value == 1}
    for card in deck.cards:
        assert card.value.char in valid_values
    assert deck.cards_left == 24  # 6 values * 4 suits


def test_hand_get_from_text_and_rank_properties():
    from gamble.models.cards import Hand

    # Straight flush: 5 hearts 5-9 suited (9H 8H 7H 6H 5H)
    straight_flush = Hand.get("9H,8H,7H,6H,5H")
    assert straight_flush.is_straight_flush is True
    assert straight_flush.rank.name == "straight flush"

    # Four of a kind: four 2s and a 5 (2H 2D 2C 2S 5H)
    four_kind = Hand.get("2H,2D,2C,2S,5H")
    assert four_kind.is_four_of_a_kind is True
    assert four_kind.rank.name == "four of a kind"

    # Full house: three 3s and two 6s (3H 3D 3C 6S 6H)
    full_house = Hand.get("3H,3D,3C,6S,6H")
    assert full_house.is_full_house is True
    assert full_house.rank.name == "full house"

    # Flush: all clubs but not straight (2C 4C 6C 8C KC)
    flush = Hand.get("2C,4C,6C,8C,KC")
    assert flush.is_flush is True
    assert flush.rank.name == "flush"

    # Straight: 5 cards in sequence different suits (5H 4C 3D 2S AH)
    straight = Hand.get("5H,4C,3D,2S,AH")
    assert straight.is_straight is True
    assert straight.rank.name == "straight"

    # Three of a kind: three 7s (7C 7D 7H 3S 4D)
    three_kind = Hand.get("7C,7D,7H,3S,4D")
    assert three_kind.is_three_of_a_kind is True
    assert three_kind.rank.name == "three of a kind"

    # Two pair: (5H 5C 9S 9D 2H)
    two_pair = Hand.get("5H,5C,9S,9D,2H")
    assert two_pair.is_two_pair is True
    assert two_pair.rank.name == "two pair"

    # One pair: (KH KD 3S 7H 9C)
    one_pair = Hand.get("KH,KD,3S,7H,9C")
    assert one_pair.is_one_pair is True
    assert one_pair.rank.name == "pair"

    # High card: no pairs (AH 3S 5D 8C 9H)
    high_card = Hand.get("AH,3S,5D,8C,9H")
    assert high_card.rank.name == "high card"

    # Length and string representation checks
    assert len(high_card) == 5
    assert str(high_card).startswith("[") and str(high_card).endswith("]")
    assert repr(high_card).startswith("<Hand[")


def test_hand_comparisons():
    from gamble.models.cards import Hand

    hand1 = Hand.get("9H,8H,7H,6H,5H")  # Straight flush
    hand2 = Hand.get("2H,2D,2C,2S,5H")  # Four of a kind
    hand3 = Hand.get("3H,3D,3C,6S,6H")  # Full house

    assert hand1 > hand2
    assert hand2 > hand3
    assert (hand3 < hand1) is True


def test_hand_vals_property():
    from gamble.models.cards import Hand

    hand = Hand.get("5H,5C,9S,9D,2H")  # Two pair
    assert hand._vals[0:2] == [2, 2]


def test_deck_str_and_repr():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s.startswith("<Deck[")
    assert s.endswith("]")
    assert s == r


def test_card_str_and_repr_consistency():
    card = Card.get("JS")
    s = str(card)
    r = repr(card)
    assert s == "J♠"
    assert r == f"<Card:{s}>"


def test_empty_deck_draw_raises():
    deck = Deck(cards=[], shuffle=False)
    with pytest.raises(IndexError):
        deck.draw()


def test_euchre_deck_has_only_24_cards():
    deck = EuchreDeck()
    assert deck.cards_left == 24
    values = set(c.value.char for c in deck.cards)
    expected_chars = {v.char for v in Card.Values.all() if v.value >= 9 or v.value == 1}
    assert values == expected_chars


def test_deck_draw_multiple_more_than_exists():
    deck = Deck(cards=[Card.get("AS"), Card.get("KD")], shuffle=False)
    # Draw more cards than in deck -> IndexError on pop
    with pytest.raises(IndexError):
        deck.draw(times=3)


def test_card_equality_with_different_object():
    card = Card.get("AS")
    assert (card == "some string") is False
