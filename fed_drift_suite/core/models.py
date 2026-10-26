"""
Shared schemas. Both layers import from here, so the number the trader sizes against and the number
the API publishes are the same object and cannot drift apart.

Pydantic rather than plain dataclasses because these are the wire format for the commercial layer and
FastAPI generates the OpenAPI spec from them — which is most of the client documentation a fund will
actually read.

A convention that matters for the honesty of the product: every returned expectancy is NET of
modelled friction and carries the sample size and out-of-sample p-value beside it. A drift number
without an `n` and a `p` beside it is marketing, not research, and a client's diligence team will
ask for both in the first meeting.
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Symbol = Literal["SPY", "QQQ", "IWM"]


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStyle(str, Enum):
    AUCTION = "auction"          # CLS on entry, OPG on exit — the fills the research priced
    MARKET = "market"            # fallback; pays the spread, flagged as degraded everywhere


class Window(str, Enum):
    IDLE = "idle"
    EVE = "eve"                  # today is FOMC eve: entry happens at the close
    DECISION = "decision"        # today is the decision day: exit happens at the open
    NONE = "none"


# --------------------------------------------------------------------------- research

class TradeRow(BaseModel):
    """One historical trade. This is the atom everything else is computed from."""
    symbol: Symbol
    decision_date: date
    eve_date: date
    entry_close: float
    exit_open: float
    gross_bp: float
    cost_bp: float
    net_bp: float
    sample: Literal["holdout", "design"]


class SampleStats(BaseModel):
    label: str = Field(description="'full', 'holdout' (<=2017) or 'design' (2018+)")
    n_trades: int
    first: date | None = None
    last: date | None = None
    gross_bp_mean: float
    cost_bp_mean: float
    net_bp_mean: float = Field(description="Average net basis points per trade, after all friction")
    net_bp_median: float
    net_bp_sd: float
    win_rate: float = Field(description="Fraction of trades with net_bp > 0")
    sharpe_per_trade: float
    sharpe_annualized: float = Field(
        description="sharpe_per_trade * sqrt(trades per year). Bounded by trade frequency: "
                    "8 trades/yr caps this near 1.2 regardless of per-trade quality.")
    trades_per_year: float
    max_drawdown_pct: float = Field(description="On the trade-sequence equity curve, not daily marks")
    t_stat_hac: float = Field(description="Newey-West, 5 lags, H0: mean net return = 0")
    permutation_p: float = Field(
        description="P(random same-size sample of nights beats this mean), 5000 draws. More honest "
                    "than a t-stat on a few hundred fat-tailed observations.")
    ann_return_on_full_capital_pct: float = Field(
        description="What this earns on capital held aside for it year-round. Small by construction: "
                    "the position exists ~8 nights a year.")
    drop_best_2_net_bp: float = Field(
        description="Mean net bp after deleting the two best trades. A robustness check: if the edge "
                    "lives in two nights, it is not an edge.")


class SymbolResearch(BaseModel):
    symbol: Symbol
    full: SampleStats
    holdout: SampleStats
    design: SampleStats


class Provenance(BaseModel):
    built_at: datetime
    price_source: str
    calendar_source: str
    n_trades: int
    n_baseline_nights: int
    first_trade: date
    last_trade: date
    holdout_end: date
    holdout_note: str
    cost_model: dict
    trade_definition: str


class ResearchMetrics(BaseModel):
    """Everything a diligence team needs to reproduce the claim, or refute it."""
    provenance: Provenance
    symbols: dict[str, SymbolResearch]
    recommended: Symbol
    recommendation_basis: str
    known_negative_results: list[str] = Field(
        description="Tests the hypothesis FAILED. Published deliberately: a research product that "
                    "reports only its wins has told you nothing about its selection process.")
    structural_limits: list[str]


# ----------------------------------------------------------------------------- signal

class NextSignal(BaseModel):
    as_of: date
    decision_date: date
    eve_date: date | None
    days_until_eve: int | None
    action: str = Field(description="What to do, in one line")
    symbol: Symbol
    entry: str = Field(description="When and how to enter")
    exit: str = Field(description="When and how to exit")
    expected_drift_bp: float = Field(description="Net expectancy per trade, full-sample")
    expected_drift_bp_holdout: float = Field(
        description="The same number on out-of-sample data only. Use this one for sizing.")
    historical_win_rate: float
    historical_max_drawdown_pct: float
    n_historical_trades: int
    permutation_p: float
    confidence: str
    caveats: list[str]


class StatusResponse(BaseModel):
    as_of: datetime
    market_date: date
    window: Window
    is_fomc_eve: bool
    is_decision_day: bool
    decision_date: date | None
    next_decision_date: date | None
    days_to_next_decision: int | None
    message: str


# ---------------------------------------------------------------------------- webhooks

class WebhookSubscribeRequest(BaseModel):
    url: HttpUrl = Field(description="HTTPS endpoint that will receive the 15:00 ET signal POST")
    symbols: list[Symbol] = Field(default=["QQQ"])
    secret: str | None = Field(
        default=None, min_length=16,
        description="Optional shared secret. If set, deliveries carry an HMAC-SHA256 signature in "
                    "X-Signature so you can verify the payload really came from us.")
    label: str | None = None

    @field_validator("url")
    @classmethod
    def must_be_https(cls, v: HttpUrl) -> HttpUrl:
        # Signals are commercially sensitive and the secret is used to sign them. Plain HTTP would
        # leak both to anyone on the path.
        if v.scheme != "https":
            raise ValueError("webhook URL must be https")
        return v


class WebhookSubscription(BaseModel):
    id: str
    url: HttpUrl
    symbols: list[Symbol]
    label: str | None = None
    signed: bool = False
    created_at: datetime
    active: bool = True
    last_delivery_at: datetime | None = None
    last_delivery_status: str | None = None
    consecutive_failures: int = 0


class WebhookDelivery(BaseModel):
    event: Literal["fomc_eve_signal", "test"]
    subscription_id: str
    sent_at: datetime
    signal: NextSignal | None = None


# ----------------------------------------------------------------------------- trading

class Fill(BaseModel):
    symbol: Symbol
    side: Side
    qty: float
    price: float | None = None
    order_id: str
    style: OrderStyle
    degraded: bool = False
    submitted_at: datetime | None = None


class ClosedTrade(BaseModel):
    symbol: Symbol
    qty: float
    entry_price: float
    exit_price: float
    cost_basis: float
    pnl: float
    pnl_pct: float
    degraded: bool = False
