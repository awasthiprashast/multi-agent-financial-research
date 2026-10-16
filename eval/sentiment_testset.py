"""
eval/sentiment_testset.py — held-out sentiment test set.

Important caveat, stated plainly because it affects how the number should be
read: these 48 sentences are hand-written and hand-labelled for this project.
They are deliberately *not* drawn from the templates used to build the snapshot
corpus, and they include the cases that separate a real financial sentiment
model from a keyword counter — negation, hedging, mixed polarity, and headlines
whose surface words point the opposite way to their financial meaning.

Measuring the lexicon fallback on the snapshot corpus would be circular (the
corpus was written with lexicon vocabulary). This set is the honest comparison.
A public benchmark run on Financial PhraseBank is listed as remaining work.

Labels follow the FinBERT convention: positive / negative / neutral, from the
perspective of the issuer's shareholders.
"""

from typing import List, Tuple

TEST_SET: List[Tuple[str, str]] = [
    # --- clearly positive ------------------------------------------------
    ("Quarterly profit climbed 18% as the services division returned to growth.", "positive"),
    ("The company secured its largest ever order, worth $1.2 billion over five years.", "positive"),
    ("Operating leverage drove margin expansion for the fourth straight quarter.", "positive"),
    ("Free cash flow conversion improved sharply on better working capital discipline.", "positive"),
    ("The board approved a higher dividend after a record year for collections.", "positive"),
    ("Management lifted full-year guidance following a strong festive quarter.", "positive"),
    ("Net debt was fully repaid ahead of schedule, leaving the balance sheet unencumbered.", "positive"),
    ("Deal wins in the banking vertical reached an eight-quarter high.", "positive"),

    # --- clearly negative ------------------------------------------------
    ("The firm cut its revenue outlook after two large clients deferred projects.", "negative"),
    ("Margins compressed as wage inflation outpaced pricing.", "negative"),
    ("The regulator opened an inquiry into the company's related-party transactions.", "negative"),
    ("A goodwill impairment of $400 million pushed the quarter into a loss.", "negative"),
    ("Attrition accelerated and the replacement cost weighed on delivery.", "negative"),
    ("The auditor flagged a material weakness in internal financial controls.", "negative"),
    ("Order inflow fell to its weakest level in three years.", "negative"),
    ("The company suspended its dividend to preserve liquidity.", "negative"),

    # --- clearly neutral -------------------------------------------------
    ("The company will report results on the fourteenth of next month.", "neutral"),
    ("The annual general meeting will be held virtually this year.", "neutral"),
    ("The stock will trade ex-dividend from Thursday.", "neutral"),
    ("The board appointed a new company secretary with effect from April.", "neutral"),
    ("Shares closed broadly unchanged in line with the benchmark index.", "neutral"),
    ("The firm filed its quarterly disclosure with the exchange as required.", "neutral"),
    ("Trading volumes matched the twenty-day average during the session.", "neutral"),
    ("The company confirmed the record date for the scheme of arrangement.", "neutral"),

    # --- negation and hedging (the hard cases) ---------------------------
    ("Growth was not as weak as the market had feared.", "positive"),
    ("The quarter was no disaster, but nor was it the inflection investors wanted.", "neutral"),
    ("Margins did not decline despite the wage revision.", "positive"),
    ("The company failed to sustain the momentum seen in the previous quarter.", "negative"),
    ("Results were not enough to offset concerns about the order pipeline.", "negative"),
    ("Guidance was maintained rather than raised, disappointing some investors.", "negative"),
    ("The impairment was smaller than analysts had modelled.", "positive"),
    ("Cost savings could not fully absorb the pricing pressure.", "negative"),

    # --- mixed polarity ---------------------------------------------------
    ("Revenue beat expectations but margins missed on higher subcontracting costs.", "neutral"),
    ("Strong deal wins were offset by a sharp rise in receivable days.", "neutral"),
    ("The company grew faster than peers, though from a much smaller base.", "neutral"),
    ("Profit rose on a one-off tax credit while the operating line deteriorated.", "negative"),
    ("Losses narrowed considerably and the path to breakeven was brought forward.", "positive"),
    ("Headcount fell but utilisation improved to a multi-year high.", "positive"),

    # --- surface words point the wrong way --------------------------------
    ("The stock fell even as the company reported its best quarter on record.", "positive"),
    ("Shares rallied on speculation the loss-making unit will be shut down.", "positive"),
    ("A large buyback was announced amid concerns over capital allocation.", "neutral"),
    ("Analysts raised concerns that the growth is being bought with discounts.", "negative"),
    ("The company denied reports of an accounting investigation.", "neutral"),
    ("Cost cutting continued, with a further reduction in the workforce announced.", "negative"),
    ("The rating agency affirmed its outlook at stable.", "neutral"),
    ("The turnaround plan has yet to show results in the reported numbers.", "negative"),
    ("Provisioning was raised prudently ahead of an expected credit cycle.", "neutral"),
    ("The subsidiary was divested at a premium to its carrying value.", "positive"),
]


def texts_and_labels():
    return [t for t, _ in TEST_SET], [l for _, l in TEST_SET]
