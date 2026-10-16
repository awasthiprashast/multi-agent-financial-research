# Manual audit of LLM-written claims

Model: Ollama `mistral:latest` (7B, local). Tickers: TCS, Reliance, Infosys (snapshot
corpus). Five sampling seeds (7, 23, 101, 555, 909), 267 claims.

The numeric oracle in `eval/llm_eval.py` checks quantities against structured source
values. It cannot see directional or interpretive errors, so the technical claims
were read by hand against the true indicator values (`TCS`: close 3202.39, SMA20
3383.46, SMA50 3627.92, RSI 23.8, MACD −109.65 vs signal −102.02; `Reliance`: close
1325.9, SMA20 1314.42, SMA50 1334.15, RSI 54.0, MACD −9.58 vs −11.26; `Infosys`: close
1512.37, SMA20 1433.31, SMA50 1421.63, RSI 66.8, MACD 11.65 vs 1.68, Bollinger upper
1511.46).

## How the verifier was developed against this data (read before trusting numbers)

The directional rules were written in rounds, each after auditing the previous
round's output, so early seeds are **in-sample**:

| Round | Found by audit | Fix |
|---|---|---|
| 1 (seed 7) | verifier flagged 30 % of claims, many wrongly: label window too narrow for long LLM sentences; unsigned drawdown/VaR vs negative values; the numeric oracle itself ignored units | clause-level labels; magnitude match for drawdown/loss; recompute always available; oracle fixed |
| 1 | Infosys "RSI above 70, overbought" (RSI 66.8) **surfaced** | `INFY.NS` dot broke the regex spans; added Bollinger / extreme / "decreased over window" rules |
| 2 (seeds 23, 101) | "higher than the 20-day average **but lower than the 50-day**" (false) **surfaced** | continuation-clause rule; average-as-subject rule |
| 3 (seeds 555, 909, never seen by the rules) | "close is **equal to** the 20-day average" (0.9 % off) **surfaced** | equality rule |

| 4 (Groq `gpt-oss-120b`) | 14 of 29 withheld claims were **correct**: "2026-01-01 to 2026-12-28" was read as the numbers −01, −01, −12, −28; "a high of ₹1493" did not match "the highest close"; "closed at" did not match "close" | dates masked before number extraction; small word-form map (highest→high, closed→close, fell→decline …) |

Seeds 555 and 909 were generated after round 2, so they were a partly held-out test;
they still revealed one gap (the round-3 rule). All five seeds were then re-run on the
final code, and the final-code surfaced technical claims were re-audited.

## Final-code audit (distinct surfaced technical claims across the five seeds)

61 distinct technical claims were surfaced.

- **0** state a checkable fact (a level, a comparison, an indicator zone, a direction
  over the window) that is wrong.
- **1** is vague and not checkable: Reliance "close is within the Bollinger band lower
  limit" (the close sits mid-band).
- **2** have a true premise and a wrong conclusion, which no rule here can detect:
  "20-day SMA below the price, suggesting a downward trend" (that suggests the
  opposite).

Not audited: fundamental, sentiment and risk prose beyond the numeric oracle.
Sentiment claims in particular include speculation ("could lead to increased
revenue") that is surfaced because it has nothing checkable.

## Summary

| | |
|---|---|
| LLM claims | 267 (fundamental 75, technical 74, sentiment 45, risk 73) |
| Withheld on first pass | 27 (10.1 %) |
| …of which direction contradicted by raw prices | 21 (28 % of the 74 technical claims) |
| …of which a number not in the evidence or recomputation | 5 |
| …of which weak semantic support | 1 |
| Claims with a wrong number per the structured oracle | 3, all withheld |
| Revisions attempted / successful | 30 / 24 |

The model rarely gets a *number* wrong when the number is in front of it; it often gets
a *direction* wrong. That is the failure the directional checks exist for.
