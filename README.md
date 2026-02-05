# Sentiment-driven-Alpha-Generation-for-Energy-ETFs

A production-grade quantitative trading system that extracts sentiment signals from 10+ years of financial news and integrates them into a walk-forward optimized strategy with convex payoff characteristics.

---

## Project Overview

This project demonstrates end-to-end development of an alternative data pipeline for systematic trading:

1. **Data Engineering**: Streaming ETL processing 45M+ news records from Reuters, Bloomberg, and AP
2. **NLP Infrastructure**: Domain-specific sentiment scoring with deduplication at scale
3. **Feature Engineering**: Market-aligned sentiment factors synchronized to trading calendars
4. **Strategy Development**: Walk-forward optimized system with trend filters, momentum locks, and derivatives hedging
5. **Risk Analytics**: Institutional-grade performance metrics and tail risk analysis

**Key Results**: 49% CAGR | 2.3 Sortino Ratio | -26% Max Drawdown (vs -70% benchmark) | 0.36 Beta

---

## Architecture

### 1. Data Pipeline (`STEP 1-2`)

**Enrichment & Normalization**
- Streaming JSON parsing with PyArrow to handle large-scale datasets
- Date filtering and field extraction from semi-structured news data
- DuckDB-powered aggregation avoiding memory bottlenecks

**Hybrid Deduplication System**
- **SimHash (LSH)**: Fast approximate matching with 64-bit fingerprints and k=3 near-duplicate detection
- **SBERT Embeddings**: Semantic similarity using `all-MiniLM-L6-v2` transformer model
- **FAISS Indexing**: GPU-accelerated cosine similarity search (threshold: 0.90)
- **Union-Find Clustering**: Efficient graph-based grouping of redundant articles

*Impact*: 40% reduction in redundancy while preserving cross-source informational diversity

### 2. NLP Scoring (`STEP 3`)

**Domain-Specific Sentiment Analysis**
- **Model**: FinBERT (ProsusAI) fine-tuned on financial text
- **Classification**: Dynamic bucketing into macro, tech, energy, and general categories
- **Inference**: GPU-accelerated batch processing (batch_size=128)
- **Output**: Sentiment scores + article volume metrics per category

**Category Keywords**:
- **Macro**: Fed, inflation, rates, CPI, treasury, recession, GDP
- **Tech**: AAPL, MSFT, NVDA, AI, semiconductors, cloud
- **Energy**: Oil, crude, natural gas, OPEC, refineries

### 3. Feature Engineering (`STEP 4`)

**Market Alignment**
- Forward-fill sentiment to next trading day (handles weekend/holiday news)
- Synchronization with S&P 500 trading calendar via `yfinance`
- Log-transformed volume features: `np.log1p(article_count)`
- Average sentiment per category: `sum_sentiment / count`

**Final Features** (per trading day):
```
sent_macro, vol_macro
sent_tech, vol_tech
sent_energy, vol_energy
sent_general, vol_general
```

### 4. Systematic Strategy

**Signal Generation**
- **Trend Filter**: 200-day moving average (uptrend gate)
- **Momentum Lock**: 45-day moving average (downside protection)
- **Sentiment Z-Scores**: Multi-window normalization (15/30/45/60/75/90/120 days)

**Position Logic**
```python
if price > MA_200:
    position = LONG_ASSET
elif (sentiment_z < threshold) AND (price < MA_45):
    position = PUT_HEDGE
else:
    position = CASH
```

**Walk-Forward Optimization**
- Training window: 2 years rolling
- Test period: 1 year forward
- Hyperparameters: sentiment threshold [-1.0 to -2.0], hedge allocation [2%-10%]
- Prevents look-ahead bias and overfitting

**Convex Hedging Layer**
- 90-day 10% OTM put options priced via Black-Scholes
- Dynamic IV and dividend yield adjustments
- Asymmetric payoff: limited downside, unlimited upside

---

## Performance Metrics

### Returns & Risk-Adjusted Performance
| Metric | Strategy | Benchmark (XLE) |
|--------|----------|-----------------|
| Total Return | **2,249%** | 26% |
| CAGR | **49.1%** | 3.0% |
| Sharpe Ratio | **1.24** | 0.17 |
| Sortino Ratio | **2.31** | — |
| Calmar Ratio | **1.91** | 0.04 |

### Tail Risk & Drawdown
| Metric | Strategy | Benchmark |
|--------|----------|-----------|
| Max Drawdown | **-25.8%** | -70.1% |
| 95% CVaR | **-3.2%** | -4.6% |
| Skewness | **16.34** | -0.50 |
| Kurtosis | **486.92** | 12.78 |

### Market Exposure
| Metric | Value |
|--------|-------|
| Beta | 0.36 |
| Alpha (Annual) | 49.6% |
| Win Rate | 33.2% |
| Profit Factor | 1.54 |

*Positive skewness and reduced beta demonstrate convex payoff structure with asymmetric tail protection.*

---

## Technical Stack

**Data Engineering**
- `DuckDB`: SQL analytics on Parquet datasets
- `PyArrow`: Zero-copy streaming I/O
- `Pandas`: Data manipulation and aggregation

**NLP & Machine Learning**
- `Transformers` (HuggingFace): FinBERT sentiment model
- `Sentence-Transformers`: SBERT embeddings
- `SimHash`: Locality-sensitive hashing
- `FAISS`: Vector similarity search
- `PyTorch`: GPU acceleration

**Quantitative Finance**
- `NumPy/SciPy`: Statistical analysis and Black-Scholes pricing
- `yfinance`: Market data and trading calendar
- Custom walk-forward optimization framework

**Visualization**
- `Matplotlib`: Performance charts and regime visualization

---

## Key Implementation Details

### Memory-Efficient Processing
```python
# Streaming dataset iteration (avoids loading 45M rows into RAM)
dataset = ds.dataset(TEMP_DIR, format="parquet")
with pq.ParquetWriter(COMBINED_FILE, dataset.schema) as writer:
    for batch in dataset.to_batches():
        writer.write_batch(batch)
```

### GPU-Accelerated Inference
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Batch processing for efficiency
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
```

### Black-Scholes Option Pricing
```python
def get_bs_put(S, K, T, r, q, sigma):
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
```

---

## File Structure

```
├── data/
│   ├── raw/                    # Original news shards (shard_*.parquet)
│   ├── temp/                   # Enriched intermediate files
│   ├── processed/
│   │   ├── combined.parquet    # Merged dataset
│   │   ├── de-duplicated.parquet
│   │   └── dedupe_temp/        # Daily deduplication shards
│   └── sentiment_chunks/       # Scored sentiment batches
│
├── sentiment_dataset.parquet  # Final feature dataset
├── Bloomberg.xlsx                  # Market data (Spot, IV, Rates)
└── sentiment_driven_strategy_XLE.ipynb            # Main analysis notebook
```

---

## Methodology Highlights

### Deduplication Strategy
The hybrid approach combines speed and accuracy:
- **SimHash**: O(1) lookup for exact/near matches → reduces 45M to ~27M
- **SBERT**: Catches paraphrased content → final ~25M unique articles
- **Representative Selection**: Longest article from most authoritative source per cluster

### Walk-Forward Robustness
Strategy parameters are re-optimized annually using only past data:

| Year | Optimal Config | Out-of-Sample Return |
|------|----------------|----------------------|
| 2018 | (60d, -2.0σ, 10%) | +30.3% |
| 2020 | (30d, -1.25σ, 10%) | +289.6% |
| 2022 | (30d, -1.25σ, 10%) | +67.5% |

*No look-ahead bias: each year's test performance uses parameters trained only on prior data.*

### Regime Detection Logic
Three-layer filter prevents false signals:
1. **Trend Gate**: Only allows long exposure in uptrends (Price > MA_200)
2. **Fear Trigger**: Activates hedging when sentiment reaches extreme negativity
3. **Momentum Lock**: Requires price weakness (Price < MA_45) to confirm hedge

This multi-condition approach reduces whipsaws and aligns hedging with actual market stress.

---

## Limitations & Future Work

**Current Limitations**:
- Single-asset backtest (XLE); diversification across sectors needed
- Transaction costs not modeled (option spreads, slippage)
- No portfolio construction or position sizing framework
- Limited to equity options (no VIX, rates, or commodity hedges)

**Potential Enhancements**:
- Multi-asset universe with correlation-based portfolio optimization
- Incorporate intraday news flow for high-frequency signal updates
- Add macro regime classification (recession, expansion, stagflation)
- Ensemble models combining FinBERT with traditional NLP (bag-of-words, TF-IDF)
- Real-time deployment with streaming news APIs

---

## References

**Dataset**:
```bibtex
@misc{brian_ferrell_2025,
    author       = { Brian Ferrell },
    title        = { financial-news-multisource (Revision b509ef6) },
    year         = 2025,
    url          = { https://huggingface.co/datasets/Brianferrell787/financial-news-multisource },
    doi          = { 10.57967/hf/6432 },
    publisher    = { Hugging Face }
}
```

**Models & Libraries**:
- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- SBERT: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- SimHash: [1e0ng/simhash](https://github.com/1e0ng/simhash)
- FAISS: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

**Academic Context**:
- Black-Scholes-Merton Option Pricing Model (1973)
- Walk-Forward Analysis for robust backtesting (Pardo, 2008)
- NLP in Finance: Sentiment Analysis and Market Prediction

---

## License

This project is licensed under the Creative Commons Zero v1.0 Universal License - see the [LICENSE](https://github.com/atharva-m/Sentiment-driven-Alpha-Generation-for-Energy-ETFs/blob/main/LICENSE) file for details.

**Disclaimer**: This software is for educational purposes only. It does not constitute financial advice. The authors are not responsible for any financial losses incurred from the use of this strategy.

**Data Attribution**: Financial news dataset sourced from [Brianferrell787/financial-news-multisource](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) (Hugging Face). The raw data is not included in this repository.

---

## Authors
- Atharva Mokashi
- Adarsh Prabhudesai
- Alina Hasan
- Sai Dinesh Devineni

---

## Contact

For questions about methodology or implementation details, please open an issue or reach out via atharvamokashi01@gmail.com, adarshprabhudesai09@gmail.com, saidevineni25@gmail.com, 1alinahasan@gmail.com
