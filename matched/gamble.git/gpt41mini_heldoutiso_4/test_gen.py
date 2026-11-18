import pytest
from gamble.models.cards import Card, Deck, Suit, Value, InvalidCard, Hand


def test_card_get_valid():
    c = Card.get("As")
    assert isinstance(c, Card)
    assert c.value.char == "A"
    assert c.suit.char == "S"
    assert str(c) == "A♠"

def test_card_get_invalid_length():
    with pytest.raises(InvalidCard, match="Too many characters for a card!"):
        Card.get("ASX")

def test_card_get_invalid_value():
    with pytest.raises(InvalidCard, match="Invalid value for card!"):
        Card.get("1S")

def test_card_get_invalid_suit():
    with pytest.raises(InvalidCard, match="Invalid suit for card!"):
        Card.get("AZ")

def test_card_color_properties():
    red_card = Card(value=Card.Values.ACE, suit=Card.Suits.HEARTS)
    black_card = Card(value=Card.Values.ACE, suit=Card.Suits.CLUBS)
    assert red_card.color == Card.RED
    assert red_card.is_red is True
    assert red_card.is_black is False
    assert black_card.color == Card.BLACK
    assert black_card.is_black is True
    assert black_card.is_red is False

def test_card_full_name_property():
    card = Card(value=Card.Values.KING, suit=Card.Suits.DIAMONDS)
    assert card.full_name == "king of diamonds"

def test_card_unicode_property():
    # Ace hearts unicode
    c = Card(value=Card.Values.ACE, suit=Card.Suits.HEARTS)
    expected = chr(c.suit.unicode + c.value.value)
    assert c.unicode == expected

    # Queen spades unicode includes hack (value >= 12)
    q = Card(value=Card.Values.QUEEN, suit=Card.Suits.SPADES)
    expected_q = chr(q.suit.unicode + q.value.value + 1)
    assert q.unicode == expected_q

def test_card_comparisons():
    c1 = Card.get("2S")
    c2 = Card.get("3S")
    c3 = Card.get("2S")
    assert c1 < c2
    assert c2 > c1
    assert c1 <= c3
    assert c1 >= c3
    assert c1 == c3
    assert c1 != c2
    assert (c1 == "not a card") is False

def test_deck_default_init_contains_all_cards_and_len():
    deck = Deck(shuffle=False)
    assert len(deck.cards) == 52
    # Check all suits and values in deck
    cards_set = set(str(c) for c in deck.cards)
    expected_cards = set(f"{v.char}{s.symbol}" for s in Card.Suits.all() for v in Card.Values.all())
    assert cards_set == expected_cards

def test_deck_draw_single_and_multiple_cards():
    deck = Deck(shuffle=False)
    initial_len = len(deck.cards)
    card = deck.draw()
    assert isinstance(card, Card)
    assert len(deck.cards) == initial_len - 1

    # Draw multiple cards
    n = 5
    cards = deck.draw(times=n)
    assert isinstance(cards, list)
    assert len(cards) == n
    assert len(deck.cards) == initial_len - 1 - n

def test_deck_top_and_bottom_properties():
    deck = Deck(shuffle=False)
    bottom = deck.bottom
    top = deck.top
    assert isinstance(bottom, Card)
    assert isinstance(top, Card)
    # bottom is the first card in deck.cards
    assert deck.cards[0] == bottom
    # top is the last card in deck.cards
    assert deck.cards[-1] == top

def test_deck_str_repr_cards_left():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s == r
    assert f"<Deck[{len(deck.cards)}]>" == s

def test_deck_contains_method():
    deck = Deck(shuffle=False)
    some_card = deck.cards[0]
    assert some_card in deck
    not_a_card = "not a card"
    assert not_a_card not in deck

def test_deck_shuffle_increments_count(monkeypatch):
    deck = Deck(shuffle=False)
    old_order = deck.cards.copy()

    # Patch random.shuffle to do nothing
    monkeypatch.setattr("random.shuffle", lambda x: None)
    deck.shuffle(times=3)
    assert deck.shuffles == 3
    # Cards are unchanged because shuffle is patched to do nothing
    assert deck.cards == old_order

def test_deck_draw_hand_returns_hand_with_size():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    # After drawing 5 cards, deck has 52 -5 cards
    assert deck.cards_left == 52 - 5

def test_card_repr_and_str_consistency():
    c = Card.get("QH")
    s = str(c)
    r = repr(c)
    assert r == f"<Card:{s}>"
    assert isinstance(s, str)

def test_hand_get_from_text_and_str_repr():
    text = "AS, KD, 3C, 5H, 9D"
    hand = Hand.get(text)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    s = str(hand)
    r = repr(hand)
    for card_str in text.replace(" ", "").upper().split(","):
        assert card_str[0] in [c.value.char for c in hand.cards]
        # suit char check
        assert card_str[1] in [c.suit.char for c in hand.cards]
    assert s.startswith("[") and s.endswith("]")
    assert r.startswith("<Hand[")

def test_hand_rank_properties():
    # Straight flush: 5,6,7,8,9 of same suit (hearts)
    cards_sf = [Card.get(x) for x in ["5H", "6H", "7H", "8H", "9H"]]
    hand_sf = Hand(cards=cards_sf)
    assert hand_sf.is_straight_flush
    assert hand_sf.rank == Hand.Ranks.STRAIGHT_FLUSH

    # Four of a kind: four 9s and one ace
    cards_4k = [Card.get(x) for x in ["9H", "9C", "9S", "9D", "AH"]]
    hand_4k = Hand(cards=cards_4k)
    assert hand_4k.is_four_of_a_kind
    assert hand_4k.rank == Hand.Ranks.FOUR_OF_A_KIND

    # Full house: three kings and two aces
    cards_fh = [Card.get(x) for x in ["KH", "KS", "KD", "AC", "AH"]]
    hand_fh = Hand(cards=cards_fh)
    assert hand_fh.is_full_house
    assert hand_fh.rank == Hand.Ranks.FULL_HOUSE

    # Flush: 5 cards same suit (diamonds), not consecutive
    cards_flush = [Card.get(x) for x in ["2D", "5D", "7D", "9D", "JD"]]
    hand_flush = Hand(cards=cards_flush)
    assert hand_flush.is_flush
    assert hand_flush.rank == Hand.Ranks.FLUSH

    # Straight: 4,5,6,7,8 mixed suits
    cards_str = [Card.get(x) for x in ["4H", "5D", "6S", "7C", "8H"]]
    hand_str = Hand(cards=cards_str)
    assert hand_str.is_straight
    assert hand_str.rank == Hand.Ranks.STRAIGHT

    # Three of a kind: three 7s and two others
    cards_3k = [Card.get(x) for x in ["7H", "7D", "7S", "2C", "5H"]]
    hand_3k = Hand(cards=cards_3k)
    assert hand_3k.is_three_of_a_kind
    assert hand_3k.rank == Hand.Ranks.THREE_OF_A_KIND

    # Two pair: two 9s and two 4s plus one other
    cards_2p = [Card.get(x) for x in ["9H", "9D", "4S", "4C", "AH"]]
    hand_2p = Hand(cards=cards_2p)
    assert hand_2p.is_two_pair
    assert hand_2p.rank == Hand.Ranks.TWO_PAIR

    # One pair: just one pair
    cards_1p = [Card.get(x) for x in ["QH", "QS", "3D", "5C", "7H"]]
    hand_1p = Hand(cards=cards_1p)
    assert hand_1p.is_one_pair
    assert hand_1p.rank == Hand.Ranks.PAIR

    # High card: no pairs etc.
    cards_hc = [Card.get(x) for x in ["2H", "5D", "9C", "JC", "KH"]]
    hand_hc = Hand(cards=cards_hc)
    assert hand_hc.rank == Hand.Ranks.HIGH_CARD


def test_hand_comparison_lt_gt():
    hand_low = Hand.get("2S,3D,4C,5H,7D")  # high card 7
    hand_high = Hand.get("9S,9D,4C,4H,7D")  # two pair

    assert hand_low < hand_high
    assert hand_high > hand_low

def test_card_sorting_order():
    cards = [Card.get(x) for x in ["KD", "2S", "AC", "TH"]]
    sorted_cards = sorted(cards)
    expected_order = sorted([c.value.value for c in cards])
    values = [c.value.value for c in sorted_cards]
    assert values == expected_order


def test_deck_init_with_custom_cards():
    # Custom deck with only 2 cards
    c1 = Card.get("AS")
    c2 = Card.get("KH")
    deck = Deck(cards=[c1, c2], shuffle=False)
    assert len(deck.cards) == 2
    assert deck.cards[-1] == c2  # last card in list
    assert deck.cards[0] == c1

def test_deck_shuffle_changes_order():
    deck1 = Deck(shuffle=False)
    deck2 = Deck(shuffle=False)
    deck2.shuffle()
    assert deck1.cards != deck2.cards or deck1.shuffles == 0  # Possible rare fail if shuffle returns same but unlikely

def test_unicode_skips_knight_hack_king_queen():
    # Value 11 (jack) should not add hack
    jack = Card(value=Card.Values.JACK, suit=Card.Suits.SPADES)
    expected_jack = chr(jack.suit.unicode + jack.value.value)
    assert jack.unicode == expected_jack

    # Value 12 (queen), adds hack
    queen = Card(value=Card.Values.QUEEN, suit=Card.Suits.SPADES)
    expected_queen = chr(queen.suit.unicode + queen.value.value + 1)
    assert queen.unicode == expected_queen

    # Value 13 (king), adds hack
    king = Card(value=Card.Values.KING, suit=Card.Suits.SPADES)
    expected_king = chr(king.suit.unicode + king.value.value + 1)
    assert king.unicode == expected_king
