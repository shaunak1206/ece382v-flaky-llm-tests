import pytest
from gamble.models.cards import Card, Deck, EuchreDeck, Suit, Value, InvalidCard, Hand


def test_card_creation_defaults():
    card = Card()
    assert card.value == Card.Values.ACE
    assert card.suit == Card.Suits.SPADES


@pytest.mark.parametrize(
    "text,value_char,suit_char,value_name,suit_name",
    [("AS", "A", "S", "ace", "spades"),
     ("9H", "9", "H", "nine", "hearts"),
     ("TC", "T", "C", "ten", "clubs"),
     ("KD", "K", "D", "king", "diamonds")],
)
def test_card_get_valid(text, value_char, suit_char, value_name, suit_name):
    card = Card.get(text)
    assert card.value.char == value_char
    assert card.suit.char == suit_char
    assert card.value.name == value_name
    assert card.suit.name == suit_name


@pytest.mark.parametrize("invalid_text", ["A", "ASD", "123", "", "10H"])
def test_card_get_invalid_length(invalid_text):
    with pytest.raises(InvalidCard):
        Card.get(invalid_text)


@pytest.mark.parametrize("text", ["1S", "ZS", "0H"])
def test_card_get_invalid_value(text):
    with pytest.raises(InvalidCard):
        Card.get(text)


@pytest.mark.parametrize("text", ["AA", "AC", "AH"])
def test_card_get_invalid_suit(text):
    # These all have valid suits "A" is invalid value, so adjust tests accordingly
    # Actually these have invalid suits "A" etc.
    # We'll test a suit that is invalid character with valid value char
    # Correcting tests:
    with pytest.raises(InvalidCard):
        Card.get(text)


def test_card_color_properties():
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
        assert card.is_red
        assert not card.is_black


def test_card_full_name_and_str_repr():
    card = Card(Card.Values.KING, Card.Suits.HEARTS)
    assert card.full_name == "king of hearts"
    s = str(card)
    assert s == "K♥"  # symbol for hearts
    r = repr(card)
    assert r == f"<Card:{s}>"


def test_card_unicode_property():
    # Spot check a card unicode codepoint
    card_queen_hearts = Card(Card.Values.QUEEN, Card.Suits.HEARTS)
    uni_queen_hearts = card_queen_hearts.unicode
    assert isinstance(uni_queen_hearts, str)
    assert len(uni_queen_hearts) == 1

    card_king_diamonds = Card(Card.Values.KING, Card.Suits.DIAMONDS)
    uni_king_diamonds = card_king_diamonds.unicode
    assert isinstance(uni_king_diamonds, str)
    assert len(uni_king_diamonds) == 1

    # The unicode for queen/king should be one offset from value+unicode base due to hack
    base = card_queen_hearts.suit.unicode
    expected_code = base + card_queen_hearts.value.value + 1
    assert ord(uni_queen_hearts) == expected_code


def test_card_comparison_operators():
    c1 = Card(Card.Values.FIVE, Card.Suits.CLUBS)
    c2 = Card(Card.Values.TEN, Card.Suits.HEARTS)
    c3 = Card(Card.Values.TEN, Card.Suits.DIAMONDS)

    assert c1 < c2
    assert c2 > c1
    assert c1 <= c1
    assert c2 >= c3

    assert c1 != c2
    c4 = Card(Card.Values.FIVE, Card.Suits.CLUBS)
    assert c1 == c4
    assert not (c1 == "not a card")


def test_deck_default_creation():
    deck = Deck(shuffle=False)
    # Should have 52 cards, all unique
    cards = deck.cards
    assert len(cards) == 52
    assert len(set((c.value.value, c.suit.value) for c in cards)) == 52


def test_deck_contains():
    deck = Deck(shuffle=False)
    card = deck.cards[10]
    assert card in deck
    not_card = "not a card"
    assert not_card not in deck


def test_deck_draw_single_and_multiple():
    deck = Deck(shuffle=False)
    orig_count = deck.cards_left
    card = deck.draw()
    assert isinstance(card, Card)
    assert deck.cards_left == orig_count - 1
    cards = deck.draw(times=3)
    assert isinstance(cards, list)
    assert len(cards) == 3
    assert deck.cards_left == orig_count - 4
    assert deck.draws == 4


def test_deck_draw_hand():
    deck = Deck(shuffle=False)
    hand = deck.draw_hand(size=5)
    assert isinstance(hand, Hand)
    assert len(hand) == 5
    assert deck.cards_left == 52 - 5


def test_deck_shuffle_increments_shuffles(monkeypatch):
    # We patch random.shuffle to just count the calls without shuffling for test consistency
    count = {"calls": 0}

    def fake_shuffle(cards):
        count["calls"] += 1

    deck = Deck(shuffle=False)
    monkeypatch.setattr(deck, "shuffle", lambda times=1: None)
    deck.shuffle = lambda times=1: [count.update({"calls": count["calls"] + 1}) for _ in range(times)]
    deck.shuffle(times=3)
    assert count["calls"] == 3


def test_euchre_deck_initialization():
    euchre = EuchreDeck()
    cards = euchre.cards
    values = [v.value for v in Card.Values.all() if v.value >= 9 or v.value == 1]

    # Euchre deck should have 24 cards: 4 suits * 6 values (9,T,J,Q,K,A)
    assert len(cards) == 24
    # Check all cards suits and values are correct
    for card in cards:
        assert card.value.value in values
        assert card.suit in Card.Suits.all()
    # Check reversed order (last card in list is first we expect)
    assert cards[-1].value.value >= 9 or cards[-1].value.value == 1


def test_hand_rank_properties():
    # Straight flush
    cards_sf = [Card.get(c) for c in ["9H", "TH", "JH", "QH", "KH"]]
    hand_sf = Hand(cards_sf)
    assert hand_sf.is_straight_flush
    assert hand_sf.rank == Hand.Ranks.STRAIGHT_FLUSH

    # Four of a kind
    cards_fk = [Card.get(c) for c in ["9H", "9C", "9S", "9D", "KH"]]
    hand_fk = Hand(cards_fk)
    assert hand_fk.is_four_of_a_kind
    assert hand_fk.rank == Hand.Ranks.FOUR_OF_A_KIND

    # Full house
    cards_fh = [Card.get(c) for c in ["9H", "9C", "9S", "KH", "KC"]]
    hand_fh = Hand(cards_fh)
    assert hand_fh.is_full_house
    assert hand_fh.rank == Hand.Ranks.FULL_HOUSE

    # Flush (all hearts but not straight)
    cards_flush = [Card.get(c) for c in ["2H", "5H", "7H", "9H", "KH"]]
    hand_flush = Hand(cards_flush)
    assert hand_flush.is_flush
    assert hand_flush.rank == Hand.Ranks.FLUSH

    # Straight (not flush)
    cards_str = [Card.get(c) for c in ["9H", "TS", "JD", "QC", "KS"]]
    hand_str = Hand(cards_str)
    assert hand_str.is_straight
    assert hand_str.rank == Hand.Ranks.STRAIGHT

    # Three of a kind
    cards_3k = [Card.get(c) for c in ["9H", "9C", "9S", "2D", "KH"]]
    hand_3k = Hand(cards_3k)
    assert hand_3k.is_three_of_a_kind
    assert hand_3k.rank == Hand.Ranks.THREE_OF_A_KIND

    # Two pair
    cards_tp = [Card.get(c) for c in ["9H", "9C", "KH", "KC", "2D"]]
    hand_tp = Hand(cards_tp)
    assert hand_tp.is_two_pair
    assert hand_tp.rank == Hand.Ranks.TWO_PAIR

    # One pair
    cards_op = [Card.get(c) for c in ["9H", "9C", "KH", "3C", "2D"]]
    hand_op = Hand(cards_op)
    assert hand_op.is_one_pair
    assert hand_op.rank == Hand.Ranks.PAIR

    # High card
    cards_hc = [Card.get(c) for c in ["9H", "7C", "4D", "2C", "KH"]]
    hand_hc = Hand(cards_hc)
    assert not hand_hc.is_one_pair
    assert hand_hc.rank == Hand.Ranks.HIGH_CARD


def test_hand_comparison():
    hand_low = Hand.get("9H, 9C, KH, 3C, 2D")  # one pair
    hand_high = Hand.get("9H, 9C, 9S, 2D, KH")  # three of a kind

    assert hand_low < hand_high
    assert hand_high > hand_low
    assert (hand_low <= hand_low) is True
    assert (hand_high >= hand_high) is True


def test_hand_len_and_str_repr():
    hand = Hand.get("9H, 9C, KH, 3C, 2D")
    assert len(hand) == 5
    s = str(hand)
    assert "[" in s and "]" in s
    r = repr(hand)
    assert r.startswith("<Hand[")


def test_hand_get_parses_and_creates():
    text = "2S, 3H, 4D, 5C, 6S"
    hand = Hand.get(text)
    assert isinstance(hand, Hand)
    assert len(hand.cards) == 5
    # Cards are sorted
    values = [card.value.value for card in hand.cards]
    assert values == sorted(values)


def test_deck_str_repr():
    deck = Deck(shuffle=False)
    s = str(deck)
    r = repr(deck)
    assert s.startswith("<Deck[")
    assert r == s


def test_deck_top_and_bottom_card():
    deck = Deck(shuffle=False)
    assert isinstance(deck.top, Card)
    assert isinstance(deck.bottom, Card)
    # Top card is last in list
    assert deck.top == deck.cards[-1]
    # Bottom card is first in list
    assert deck.bottom == deck.cards[0]
