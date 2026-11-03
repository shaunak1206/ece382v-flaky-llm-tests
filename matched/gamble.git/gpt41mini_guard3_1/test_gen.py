import random
import pytest
from gamble.models.cards import Card, Deck, Hand
from gamble.errors import InvalidCard


def test_card_suits_and_values_are_ordered_and_dict_consistent():
    # Suit.all returns suits sorted by value
    suits = Card.Suits.all()
    suit_values = [s.value for s in suits]
    assert suit_values == sorted(suit_values)
    # Suit.dict keys correspond to suits' chars and values correspond back to suits
    d = Card.Suits.dict()
    for suit in suits:
        assert d[suit.char] == suit

    # Values.all returns values sorted by value
    values = Card.Values.all()
    value_values = [v.value for v in values]
    assert value_values == sorted(value_values)
    # Values.dict keys correspond to values' chars and values correspond back
    vd = Card.Values.dict()
    for v in values:
        assert vd[v.char] == v


def test_card_init_defaults_and_attributes():
    c = Card()
    assert c.value == Card.Values.ACE
    assert c.suit == Card.Suits.SPADES
    # Color corresponds to suit color
    assert c.color == c.suit.color
    # is_black and is_red correspond correctly
    assert c.is_black == (c.color == Card.BLACK)
    assert c.is_red == (c.color == Card.RED)
    # full_name contains value name and suit name
    assert c.full_name == f"{c.value.name} of {c.suit.name}"
    # unicode is a single character string
    uni = c.unicode
    assert isinstance(uni, str)
    assert len(uni) == 1
    # __str__ returns value.char + suit.symbol
    assert str(c) == f"{c.value.char}{c.suit.symbol}"
    # __repr__ includes the str representation
    rep = repr(c)
    assert rep.startswith("<Card:")
    assert str(c) in rep


def test_card_unicode_is_correct_for_queen_and_king_and_others():
    # For cards value below 12 (queen), no hack applied, for >=12 hack=1
    # Test queen
    queen = Card(Card.Values.QUEEN, Card.Suits.SPADES)
    expected_queen = chr(queen.suit.unicode + queen.value.value + 1)
    assert queen.unicode == expected_queen
    # Test king
    king = Card(Card.Values.KING, Card.Suits.SPADES)
    expected_king = chr(king.suit.unicode + king.value.value + 1)
    assert king.unicode == expected_king
    # Test jack (value=11), no hack
    jack = Card(Card.Values.JACK, Card.Suits.SPADES)
    expected_jack = chr(jack.suit.unicode + jack.value.value)
    assert jack.unicode == expected_jack


def test_card_get_valid_and_invalid(monkeypatch):
    # Seed random if used in Card.get (not used but per instructions seed anyway)
    random.seed(12345)

    # Valid inputs
    valid_inputs = [
        "AS", "2C", "TD", "QH",
        "as", "tD"
    ]
    for txt in valid_inputs:
        c = Card.get(txt)
        assert isinstance(c, Card)
        assert txt[0].upper() in Card.Values.dict()
        assert txt[1].upper() in Card.Suits.dict()
        # Retrieving the same card object via individual attributes matches
        val = Card.Values.dict()[txt[0].upper()]
        suit = Card.Suits.dict()[txt[1].upper()]
        assert c.value == val
        assert c.suit == suit

    # Invalid inputs
    with pytest.raises(InvalidCard):
        Card.get("ASQ")  # too many chars
    with pytest.raises(InvalidCard):
        Card.get("A")    # too few chars
    with pytest.raises(InvalidCard):
        Card.get("1S")   # invalid value char
    with pytest.raises(InvalidCard):
        Card.get("AZ")   # invalid suit char


def test_card_comparisons():
    # Create cards with different values for comparison
    c2 = Card(Card.Values.TWO, Card.Suits.SPADES)
    c3 = Card(Card.Values.THREE, Card.Suits.CLUBS)
    c2_heart = Card(Card.Values.TWO, Card.Suits.HEARTS)

    # __lt__, __gt__, __le__, __ge__, __eq__
    assert c2 < c3
    assert c3 > c2
    assert c2 <= c3
    assert c3 >= c2
    assert c2 <= c2_heart
    assert c2 >= c2_heart
    assert c2 == c2_heart  # equal if value and suit same? No, suits differ
    # The suit differs, so equals should be False
    assert (c2 == c2_heart) is False
    # Equality with non-Card returns False
    assert (c2 == "not a card") is False


def test_deck_default_initialization_and_len_properties():
    random.seed(42)  # seed random for shuffle that happens in init
    deck = Deck()
    # A default deck has 52 cards
    assert len(deck.cards) == 52
    # cards_left matches count of cards
    assert deck.cards_left == 52
    # top and bottom are Cards and different
    assert isinstance(deck.top, Card)
    assert isinstance(deck.bottom, Card)
    assert deck.top != deck.bottom
    # string representation contains the cards left count
    s = str(deck)
    assert str(deck.cards_left) in s
    repr_ = repr(deck)
    assert repr_ == s


def test_deck_contains_and_draw_methods(monkeypatch):
    random.seed(101)
    deck = Deck(shuffle=False)
    # Create a card known to be in the deck
    card = Card.get("AS")
    assert card in deck
    # Non-card is not in deck
    assert "AS" not in deck
    # Draw one card reduces cards count by one
    initial_count = deck.cards_left
    drawn = deck.draw()
    assert isinstance(drawn, Card)
    assert deck.cards_left == initial_count - 1
    # draw times=3 returns list of 3 cards
    drawn_cards = deck.draw(times=3)
    assert isinstance(drawn_cards, list)
    assert len(drawn_cards) == 3
    assert all(isinstance(c, Card) for c in drawn_cards)
    assert deck.draws == 4  # 1 + 3 draws total


def test_deck_draw_hand_and_shuffle(monkeypatch):
    random.seed(202)
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    # Cards left decreases accordingly
    assert deck.cards_left == 52 - 5
    # Shuffling changes order but does not change cards count
    old_cards = list(deck.cards)
    random.seed(777)
    deck.shuffle(times=2)
    assert deck.shuffles == 2
    assert deck.cards_left == 52 - 5
    # Cards content must be all the same cards but likely reordered
    assert sorted(deck.cards, key=lambda c: (c.suit.value, c.value.value)) == \
           sorted(old_cards, key=lambda c: (c.suit.value, c.value.value))


def test_hand_rank_properties_for_various_hands():
    # Straight Flush: 5H 6H 7H 8H 9H
    sf_hand = Hand.get("5H,6H,7H,8H,9H")
    assert sf_hand.is_straight_flush
    assert sf_hand.rank.name == "straight flush"

    # Four of a Kind: 9H 9S 9D 9C 2H
    fk_hand = Hand.get("9H,9S,9D,9C,2H")
    assert fk_hand.is_four_of_a_kind
    assert fk_hand.rank.name == "four of a kind"

    # Full House: 8D 8S 8C 7H 7D
    fh_hand = Hand.get("8D,8S,8C,7H,7D")
    assert fh_hand.is_full_house
    assert fh_hand.rank.name == "full house"

    # Flush: 2H 5H 7H 9H KH
    f_hand = Hand.get("2H,5H,7H,9H,KH")
    assert f_hand.is_flush
    assert not f_hand.is_straight_flush
    assert f_hand.rank.name == "flush"

    # Straight: 4C 5H 6D 7S 8H
    st_hand = Hand.get("4C,5H,6D,7S,8H")
    assert st_hand.is_straight
    assert st_hand.rank.name == "straight"

    # Three of a Kind: 3S 3H 3D 7C 9H
    tk_hand = Hand.get("3S,3H,3D,7C,9H")
    assert tk_hand.is_three_of_a_kind
    assert tk_hand.rank.name == "three of a kind"

    # Two Pair: 5C 5D 9H 9S KH
    tp_hand = Hand.get("5C,5D,9H,9S,KH")
    assert tp_hand.is_two_pair
    assert tp_hand.rank.name == "two pair"

    # One Pair: 6C 6D 3H 8S QH
    op_hand = Hand.get("6C,6D,3H,8S,QH")
    assert op_hand.is_one_pair
    assert op_hand.rank.name == "pair"

    # High Card: 2C 5D 7H 9S JD
    hc_hand = Hand.get("2C,5D,7H,9S,JD")
    assert not hc_hand.is_one_pair
    assert not hc_hand.is_two_pair
    assert not hc_hand.is_three_of_a_kind
    assert not hc_hand.is_straight
    assert not hc_hand.is_flush
    assert hc_hand.rank.name == "high card"


def test_hand_len_and_str_and_repr():
    hand = Hand.get("AS,KH,QC,JD,TS")
    # len matches number of cards
    assert len(hand) == 5
    # str outputs card strings separated by commas and in square brackets
    s = str(hand)
    for card_str in ["AS", "KH", "QC", "JD", "TS"]:
        assert card_str in s
    assert s.startswith("[") and s.endswith("]")
    # repr contains size and rank name and str()
    r = repr(hand)
    assert f"<Hand[{hand.size}]" in r
    assert hand.rank.name in r
    assert str(hand) in r


def test_deck_init_with_custom_cards_and_no_shuffle():
    # create a small custom deck
    cards = [Card(Card.Values.ACE, Card.Suits.SPADES), Card(Card.Values.KING, Card.Suits.HEARTS)]
    deck = Deck(cards=cards, shuffle=False)
    assert deck.cards == cards
    assert deck.cards_left == len(cards)
    # shuffles count should be zero since shuffle=False
    assert deck.shuffles == 0


def test_euchre_deck_has_24_cards_and_correct_values():
    deck = Deck()
    # EuchreDeck uses Cards 9,10(J=11),Q=12,K=13,A=1 of each suit, total 24 cards
    from gamble.models.cards import EuchreDeck
    euchre = EuchreDeck()
    # Must have 24 cards exactly
    assert euchre.cards_left == 24
    # Every card is either in the euchre values set or not
    allowed_values = {1, 9, 10, 11, 12, 13}
    for card in euchre.cards:
        assert card.value.value in allowed_values
    # suits are all present in euchre deck cards
    suits_present = {card.suit for card in euchre.cards}
    assert suits_present == set(Card.Suits.all())
    # Cards are reversed in order upon creation
    first_card = euchre.cards[0]
    last_card = euchre.cards[-1]
    # First card should be the last suit in sorted order and last value in euchre values
    last_suit = sorted(Card.Suits.all(), key=lambda s: s.value)[-1]
    last_value = max(allowed_values)
    # Because cards are reversed from order of suits * values ascending,
    # the first card in cards should be highest suit * lowest euchre value
    # Actually, cards appended in loops suit outer, value inner, then reversed
    # Thus the first card is last value of last suit after reverse
    assert first_card.suit == last_suit
    assert first_card.value.value == min(allowed_values)
    # Last card is first value of first suit
    first_suit = sorted(Card.Suits.all(), key=lambda s: s.value)[0]
    assert last_card.suit == first_suit
    assert last_card.value.value == max(allowed_values)
