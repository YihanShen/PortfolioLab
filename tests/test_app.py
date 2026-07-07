from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from portfoliolab.app import (
    DEFAULT_BACKFILL_CODE,
    DEFAULT_REVERSION_CODE,
    INDEX_HTML,
    PortfolioLabHandler,
    _catalog_payload,
    _delete_strategy_payload,
    _delete_neutralizer_payload,
    _neutralization_library_payload,
    _neutralization_groups,
    _lookup_symbols_payload,
    _load_initial_catalog,
    _load_price_data_payload,
    _preprocessing_library_payload,
    _refresh_catalog_payload,
    _refresh_data_payload,
    _run_payload_backtest,
    _save_preprocessor_payload,
    _save_neutralizer_payload,
    _save_strategy_payload,
    _stocks_from_payload,
    _strategy_library_payload,
)
from portfoliolab.data import Bar, MarketData
from portfoliolab.providers import fetch_nasdaq_us_symbol_catalog
from portfoliolab.strategies import load_inline_neutralizer, load_inline_strategy


class AppPayloadTest(TestCase):
    def test_catalog_and_backtest_payloads(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        pool = next(item for item in catalog["pools"] if item["id"] == "global_quality")

        result = _run_payload_backtest(
            data,
            {
                "symbols": pool["symbols"],
                "weights": {symbol: 1 / len(pool["symbols"]) for symbol in pool["symbols"]},
                "mode": "equal",
                "start": catalog["defaults"]["start"],
                "end": catalog["defaults"]["end"],
                "splitDate": catalog["defaults"]["splitDate"],
                "rebalanceFrequency": "monthly",
            },
        )

        self.assertGreaterEqual(len(catalog["stocks"]), 10)
        pool_names = [pool["name"] for pool in catalog["pools"]]
        self.assertEqual(len(pool_names), len(set(pool_names)))
        self.assertGreater(len(result["equity"]), 1_000)
        self.assertIn("full", result["metrics"])
        self.assertIn("prices", result)
        self.assertIn("capital", result)
        self.assertIn("dailyPnl", result["equity"][0])
        self.assertIn("dollarsTraded", result["equity"][0])
        self.assertIn("information_ratio", result["metrics"]["full"])
        self.assertIn("avgAbsWeight", result["stocks"][0])
        self.assertIn("avgSignedWeight", result["stocks"][0])
        self.assertIn("activeDays", result["stocks"][0])
        self.assertIn("longDays", result["stocks"][0])
        self.assertIn("shortDays", result["stocks"][0])
        self.assertTrue(all(row["year"] != "All" for row in result["yearly"]))
        self.assertTrue(all(isinstance(row["year"], int) for row in result["yearly"]))

    def test_catalog_includes_ohlc_bars_for_research_charts(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        bars = catalog["rawBars"]["AAPL"]

        self.assertGreater(len(bars), 1_000)
        self.assertEqual({"date", "open", "high", "low", "close"}, set(bars[0]))
        self.assertEqual(data.dates[-1].isoformat(), catalog["defaults"]["end"])
        defaults = catalog["defaults"]
        start = date.fromisoformat(defaults["start"])
        end = date.fromisoformat(defaults["end"])
        split = date.fromisoformat(defaults["splitDate"])
        backtest_dates = data.window_dates(start, end)
        split_position = sum(day <= split for day in backtest_dates) / len(backtest_dates)
        self.assertGreater(split_position, 0.78)
        self.assertLess(split_position, 0.82)
        self.assertIn('select id="researchWindow"', INDEX_HTML)
        self.assertIn('<option value="raw">Price</option>', INDEX_HTML)
        self.assertIn("researchRangeStart", INDEX_HTML)
        self.assertIn("researchRangeEnd", INDEX_HTML)
        self.assertIn("researchRangeStartDate", INDEX_HTML)
        self.assertIn("researchRangeEndDate", INDEX_HTML)
        self.assertIn("showSplit: false", INDEX_HTML)
        self.assertIn("loadPriceDataButton", INDEX_HTML)
        self.assertIn("Refresh Price Data", INDEX_HTML)
        self.assertIn("symbolLookupInput", INDEX_HTML)
        self.assertIn("addSymbolsButton", INDEX_HTML)
        self.assertIn("data-available-symbol", INDEX_HTML)
        self.assertIn("data-remove-available", INDEX_HTML)
        self.assertIn("removeAvailableStock", INDEX_HTML)
        self.assertIn("data-portfolio-symbol", INDEX_HTML)
        self.assertIn("/api/lookup-symbols", INDEX_HTML)
        self.assertIn("/api/load-price-data", INDEX_HTML)
        self.assertIn("dataSource", INDEX_HTML)
        self.assertIn("Yahoo Finance / yfinance", INDEX_HTML)
        self.assertIn("chart-legend-strip", INDEX_HTML)
        self.assertIn("chart-legend-viewport", INDEX_HTML)
        self.assertIn("chart-frame", INDEX_HTML)
        self.assertIn("scrollbar-width: none", INDEX_HTML)
        self.assertIn("Backtesting Results", INDEX_HTML)
        self.assertIn("results-stack", INDEX_HTML)
        self.assertIn("backtestStatus", INDEX_HTML)
        self.assertIn("backtest-action", INDEX_HTML)
        self.assertIn("compact-title", INDEX_HTML)
        self.assertIn("compact-stack", INDEX_HTML)
        self.assertIn("margin-top: 8px", INDEX_HTML)
        self.assertIn('aria-label="Signal strategy"', INDEX_HTML)
        self.assertIn("/api/strategies", INDEX_HTML)
        self.assertIn("strategy-panel", INDEX_HTML)
        self.assertIn("preprocessing-panel", INDEX_HTML)
        self.assertIn("results-grid", INDEX_HTML)
        self.assertIn('<section class="results-grid">', INDEX_HTML)
        self.assertIn("minmax(520px, 0.95fr)", INDEX_HTML)
        self.assertIn("height: 500", INDEX_HTML)
        self.assertIn("grid-template-columns: 1fr", INDEX_HTML)
        self.assertIn("strategyName", INDEX_HTML)
        self.assertIn("dataPreprocessor", INDEX_HTML)
        self.assertIn("preprocessorName", INDEX_HTML)
        self.assertIn("preprocessorCode", INDEX_HTML)
        self.assertIn("Data Preprocessing", INDEX_HTML)
        self.assertIn("Strategy Construction", INDEX_HTML)
        self.assertIn("savePreprocessorButton", INDEX_HTML)
        self.assertIn("savePreprocessorAsButton", INDEX_HTML)
        self.assertIn("deletePreprocessorButton", INDEX_HTML)
        self.assertIn("/api/preprocessors", INDEX_HTML)
        self.assertNotIn("<h2>Signal Strategy</h2>", INDEX_HTML)
        self.assertNotIn("Signal Construction", INDEX_HTML)
        self.assertIn("saveStrategyButton", INDEX_HTML)
        self.assertIn("saveStrategyAsButton", INDEX_HTML)
        self.assertIn("deleteStrategyButton", INDEX_HTML)
        self.assertIn("strategyNameExists", INDEX_HTML)
        self.assertIn('signalStrategy: "custom"', INDEX_HTML)
        self.assertIn("neutralizerCode", INDEX_HTML)
        self.assertIn("Grouping And Neutralization Code", INDEX_HTML)
        self.assertIn("selectedNeutralizationMode", INDEX_HTML)
        self.assertIn("/api/neutralizers", INDEX_HTML)
        self.assertIn(".strategy-panel textarea", INDEX_HTML)
        self.assertIn("#preprocessorCode", INDEX_HTML)
        self.assertIn("min-height: 420px", INDEX_HTML)
        self.assertIn("#portfolioChart .chart-frame > svg", INDEX_HTML)
        self.assertIn("axisFontSize: 15", INDEX_HTML)
        self.assertIn("splitFontSize: 13", INDEX_HTML)
        self.assertIn("portfolioRangeStartDate", INDEX_HTML)
        self.assertIn("portfolioRangeEndDate", INDEX_HTML)
        self.assertNotIn("Portfolio Value", INDEX_HTML)
        self.assertNotIn('value="value"', INDEX_HTML)
        self.assertIn("Portfolio PnL", INDEX_HTML)
        self.assertIn('value="sharpe">Sharpe', INDEX_HTML)
        self.assertIn('value="turnover">Turnover', INDEX_HTML)
        self.assertIn("expandingSharpeSeries", INDEX_HTML)
        self.assertNotIn("Rolling Sharpe", INDEX_HTML)
        self.assertNotIn("Rolling Turnover", INDEX_HTML)
        self.assertIn("displayPortfolioStocks", INDEX_HTML)
        self.assertIn("displayAllResearchStocks", INDEX_HTML)
        self.assertIn("clearResearchStocks", INDEX_HTML)
        self.assertIn("research-name-col", INDEX_HTML)
        self.assertIn("research-portfolio-col", INDEX_HTML)
        self.assertIn("research-industry-col", INDEX_HTML)
        self.assertIn("research-weight-col", INDEX_HTML)
        self.assertIn("Industry", INDEX_HTML)
        self.assertIn("Portfolio", INDEX_HTML)
        self.assertIn("Weight %", INDEX_HTML)
        self.assertNotIn("research-sector-col", INDEX_HTML)
        self.assertNotIn("Displayed · ${allCount}", INDEX_HTML)
        self.assertIn("stockResearchList", INDEX_HTML)
        self.assertIn("Avg Abs Weight", INDEX_HTML)
        self.assertIn("Avg Signed Weight", INDEX_HTML)
        self.assertIn("Active Days", INDEX_HTML)
        self.assertIn("Long Days", INDEX_HTML)
        self.assertIn("Short Days", INDEX_HTML)
        self.assertNotIn("Days Held", INDEX_HTML)
        self.assertLess(INDEX_HTML.find("stockResearchList"), INDEX_HTML.find("researchChartType"))
        self.assertLess(INDEX_HTML.find('<div id="preprocessingBox"'), INDEX_HTML.find('<div id="codeBox"'))
        self.assertLess(INDEX_HTML.find('<div id="codeBox"'), INDEX_HTML.find('<section class="backtest-action">'))
        self.assertLess(INDEX_HTML.find('<section class="backtest-action">'), INDEX_HTML.find('<section class="results-grid">'))
        self.assertIn("portfolioChartOptions", INDEX_HTML)
        self.assertIn("preserveResearchScroll", INDEX_HTML)
        self.assertNotIn("Backtest period:", INDEX_HTML)
        self.assertNotIn("slice(0, 6)", INDEX_HTML)
        self.assertNotIn("Demo Data", INDEX_HTML)
        self.assertNotIn("Raw Return", INDEX_HTML)
        self.assertNotIn("Window Return", INDEX_HTML)
        self.assertNotIn("Price Return", INDEX_HTML)
        self.assertNotIn("Raw Prices", INDEX_HTML)
        self.assertNotIn("<label>Signal", INDEX_HTML)
        self.assertNotIn("Refresh Catalog", INDEX_HTML)
        self.assertNotIn("Search Catalog", INDEX_HTML)
        self.assertNotIn("data-detail", INDEX_HTML)
        self.assertIn("setNeutralizerStatus(\"\")", INDEX_HTML)
        self.assertIn("setNeutralizerStatus(result.message)", INDEX_HTML)

    def test_yahoo_market_data_refresh_payload(self):
        data = MarketData.demo()
        with patch("portfoliolab.app.fetch_yahoo_data", return_value=data), patch("portfoliolab.app.write_market_data_csv"):
            result = _refresh_data_payload({"source": "yahoo", "symbols": ["AAPL", "MSFT"]})

        self.assertEqual("Yahoo Finance", result["source"])
        self.assertGreater(len(result["data"].dates), 1_000)
        self.assertIn("AAPL", result["data"].symbols)

    def test_catalog_refresh_uses_lightweight_symbol_metadata(self):
        rows = [
            {
                "symbol": "ABC",
                "provider_symbol": "ABC",
                "name": "ABC Corp",
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": "",
                "exchange": "NASDAQ",
            }
        ]
        with patch("portfoliolab.app.fetch_nasdaq_us_symbol_catalog", return_value=rows), patch("portfoliolab.app._write_catalog_cache"):
            result = _refresh_catalog_payload({"source": "yahoo"})

        self.assertEqual("Yahoo Finance", result["source"])
        self.assertEqual("ABC", result["stocks"][0].symbol)

    def test_seed_metadata_overrides_old_listing_catalog_cache(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                '{"stocks": [{"symbol": "BABA", "name": "Alibaba ADR", "region": "United States", "source": "Yahoo Finance", "sector": "", "exchange": "NYSE", "provider_symbol": "BABA"}]}',
                encoding="utf-8",
            )
            with patch("portfoliolab.app.CATALOG_CACHE_PATH", path):
                catalog = _load_initial_catalog()

        baba = next(stock for stock in catalog if stock.symbol == "BABA")
        self.assertEqual("China", baba.region)

    def test_manual_symbol_lookup_and_price_load_accepts_added_names(self):
        rows = [
            {
                "symbol": "ZZZ",
                "provider_symbol": "ZZZ",
                "name": "ZZZ Corp",
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": "Technology",
                "industry": "Software",
                "exchange": "NASDAQ",
            }
        ]
        data = MarketData([Bar(date(2024, 1, 2), "ZZZ", 10, 11, 9, 10.5, 1000)])
        with patch("portfoliolab.app.lookup_yahoo_symbols", return_value=rows):
            lookup = _lookup_symbols_payload({"source": "yahoo", "symbols": "ZZZ"})
        with patch("portfoliolab.app.fetch_yahoo_data", return_value=data), patch("portfoliolab.app.write_market_data_csv"):
            result = _load_price_data_payload({"source": "yahoo", "symbols": ["ZZZ"], "stocks": rows}, [])

        self.assertEqual("ZZZ", lookup["stocks"][0]["symbol"])
        self.assertEqual("Software", lookup["stocks"][0]["industry"])
        self.assertIn("ZZZ", result["data"].symbols)

    def test_price_refresh_prunes_removed_available_symbols(self):
        existing = MarketData([Bar(date(2024, 1, 2), "OLD", 20, 21, 19, 20.5, 1000)])
        refreshed = MarketData([Bar(date(2024, 1, 2), "NEW", 10, 11, 9, 10.5, 1000)])
        rows = [
            {
                "symbol": "NEW",
                "provider_symbol": "NEW",
                "name": "New Corp",
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": "Technology",
                "industry": "Software",
                "exchange": "NASDAQ",
            }
        ]
        with patch("portfoliolab.app.fetch_yahoo_data", return_value=refreshed), patch("portfoliolab.app.write_market_data_csv"):
            result = _load_price_data_payload({"source": "yahoo", "symbols": ["NEW"], "stocks": rows}, [], existing)

        self.assertEqual(["NEW"], result["data"].symbols)
        self.assertNotIn("OLD", result["data"].symbols)

    def test_price_refresh_requires_explicit_available_symbols(self):
        with self.assertRaisesRegex(Exception, "Add at least one available stock"):
            _load_price_data_payload({"source": "yahoo", "symbols": []}, [], None)

    def test_symbol_catalog_filters_non_common_equity_rows(self):
        nasdaq_rows = [
            {
                "Symbol": "ASML",
                "Security Name": "ASML Holding N.V. - New York Registry Shares",
                "ETF": "N",
                "Test Issue": "N",
            }
        ]
        other_rows = [
            {
                "ACT Symbol": "ALL$H",
                "Security Name": "Allstate Corporation (The) Depositary Shares each representing a 1/1,000th interest in a share of Fixed Rate Noncumulative Perpetual Preferred Stock, Series H",
                "ETF": "N",
                "Test Issue": "N",
                "Exchange": "N",
            },
            {
                "ACT Symbol": "SPY",
                "Security Name": "SPDR S&P 500 ETF Trust",
                "ETF": "Y",
                "Test Issue": "N",
                "Exchange": "P",
            },
            {
                "ACT Symbol": "BRK/B",
                "Security Name": "Berkshire Hathaway Inc.",
                "ETF": "N",
                "Test Issue": "N",
                "Exchange": "N",
            },
        ]
        with patch("portfoliolab.providers._read_symbol_directory", side_effect=[nasdaq_rows, other_rows]):
            catalog = fetch_nasdaq_us_symbol_catalog()

        symbols = {item["symbol"] for item in catalog}
        self.assertIn("ASML", symbols)
        self.assertIn("BRK/B", symbols)
        self.assertNotIn("ALL$H", symbols)
        self.assertNotIn("SPY", symbols)
        brk = next(item for item in catalog if item["symbol"] == "BRK/B")
        self.assertEqual("BRK-B", brk["provider_symbol"])

    def test_strategy_library_is_file_backed_and_rejects_duplicate_names(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "strategy_library.json"
            with patch("portfoliolab.app.STRATEGY_LIBRARY_PATH", path):
                library = _strategy_library_payload()
                self.assertGreaterEqual(len(library["strategies"]), 3)
                by_id = {item["id"]: item for item in library["strategies"]}
                self.assertIn("data.to_frame()", by_id["none"]["code"])
                self.assertIn("raw_alpha", by_id["momentum"]["code"])
                self.assertIn("relative positions", by_id["reversion"]["code"])
                self.assertEqual(str(path), library["path"])

                saved = _save_strategy_payload(
                    {
                        "name": "My Signal",
                        "code": "class Strategy:\n    def compute(self, data, as_of, context):\n        return BASE_WEIGHTS\n",
                        "saveAsNew": True,
                    }
                )
                self.assertTrue(path.exists())
                self.assertEqual("my_signal", saved["selectedId"])

                with self.assertRaises(ValueError):
                    _save_strategy_payload(
                        {
                            "name": "My Signal",
                            "code": "class Strategy:\n    def compute(self, data, as_of, context):\n        return {}\n",
                            "saveAsNew": True,
                        }
                    )

                deleted = _delete_strategy_payload({"id": "my_signal"})
                self.assertNotIn("my_signal", {strategy["id"] for strategy in deleted["strategies"]})

    def test_preprocessing_library_is_file_backed_and_validates_code(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preprocessing_library.json"
            with patch("portfoliolab.app.PREPROCESSING_LIBRARY_PATH", path):
                library = _preprocessing_library_payload()
                self.assertGreaterEqual(len(library["preprocessors"]), 3)
                by_id = {item["id"]: item for item in library["preprocessors"]}
                self.assertIn("backfill", by_id)
                self.assertIn("data.to_frame()", by_id["backfill"]["code"])
                self.assertIn("pd.MultiIndex.from_product", by_id["backfill"]["code"])
                self.assertIn("MarketData.from_frame", by_id["backfill"]["code"])
                self.assertEqual(str(path), library["path"])

                saved = _save_preprocessor_payload(
                    {
                        "name": "My Preprocess",
                        "code": "def preprocess(data, symbols, context):\n    return data\n",
                        "saveAsNew": True,
                    }
                )
                self.assertTrue(path.exists())
                self.assertEqual("my_preprocess", saved["selectedId"])

                with self.assertRaises(ValueError):
                    _save_preprocessor_payload(
                        {
                            "name": "Broken",
                            "code": "def nope(data, symbols, context):\n    return data\n",
                            "saveAsNew": True,
                        }
                    )

    def test_backfill_preprocessing_fills_missing_selected_bars(self):
        data = MarketData(
            [
                Bar(date(2024, 1, 2), "AAPL", 10, 10, 10, 10, 100),
                Bar(date(2024, 1, 3), "AAPL", 11, 11, 11, 11, 100),
                Bar(date(2024, 1, 3), "MSFT", 20, 20, 20, 20, 100),
                Bar(date(2024, 1, 4), "AAPL", 12, 12, 12, 12, 100),
                Bar(date(2024, 1, 4), "MSFT", 21, 21, 21, 21, 100),
                Bar(date(2024, 1, 5), "AAPL", 13, 13, 13, 13, 100),
                Bar(date(2024, 1, 5), "MSFT", 22, 22, 22, 22, 100),
            ]
        )
        result = _run_payload_backtest(
            data,
            {
                "symbols": ["AAPL", "MSFT"],
                "weights": {"AAPL": 0.5, "MSFT": 0.5},
                "portfolioMode": "equal",
                "preprocessingCode": DEFAULT_BACKFILL_CODE,
                "signalStrategy": "none",
                "neutralizationMode": "none",
                "start": "2024-01-02",
                "end": "2024-01-05",
                "splitDate": "2024-01-03",
                "rebalanceFrequency": "daily",
            },
        )

        self.assertEqual("2024-01-02", result["prices"]["MSFT"][0]["date"])
        self.assertGreater(sum(abs(weight) for weight in result["finalWeights"].values()), 0)

    def test_neutralization_library_is_file_backed_and_validates_code(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "neutralization_library.json"
            with patch("portfoliolab.app.NEUTRALIZATION_LIBRARY_PATH", path):
                library = _neutralization_library_payload()
                self.assertGreaterEqual(len(library["neutralizers"]), 4)
                self.assertIn("region", {item["id"] for item in library["neutralizers"]})
                self.assertIn("sector", {item["id"] for item in library["neutralizers"]})
                self.assertNotIn("subindustry", {item["id"] for item in library["neutralizers"]})
                by_id = {item["id"]: item for item in library["neutralizers"]}
                self.assertIn("def group_for", by_id["market"]["code"])
                self.assertIn("metadata row from Available Stocks", by_id["market"]["code"])
                self.assertIn("Return any label", by_id["market"]["code"])
                self.assertIn("within-group relative bet", by_id["market"]["code"])
                self.assertIn('return "Market"', by_id["market"]["code"])
                self.assertIn("return region", by_id["region"]["code"])
                self.assertIn("return sector", by_id["sector"]["code"])
                self.assertIn("return industry", by_id["industry"]["code"])
                self.assertEqual(str(path), library["path"])

                saved = _save_neutralizer_payload(
                    {
                        "name": "My Neutralizer",
                        "code": "def neutralize(alpha, groups):\n    return dict(alpha)\n",
                        "saveAsNew": True,
                    }
                )
                self.assertTrue(path.exists())
                self.assertEqual("my_neutralizer", saved["selectedId"])

                with self.assertRaises(ValueError):
                    _save_neutralizer_payload(
                        {
                            "name": "Broken",
                            "code": "def nope(alpha, groups):\n    return alpha\n",
                            "saveAsNew": True,
                        }
                    )

                deleted = _delete_neutralizer_payload({"id": "my_neutralizer"})
                self.assertNotIn("my_neutralizer", {item["id"] for item in deleted["neutralizers"]})

    def test_custom_code_sandboxes_expose_dataframe_helpers(self):
        data = MarketData.demo()
        strategy = load_inline_strategy(
            "class Strategy:\n"
            "    def compute(self, data, as_of, context):\n"
            "        frame = data.to_frame()\n"
            "        if pd is not None and not frame.empty:\n"
            "            return dict(BASE_WEIGHTS)\n"
            "        return {}\n",
            ["AAPL", "MSFT"],
            {"AAPL": 0.5, "MSFT": 0.5},
        )
        weights = strategy.compute(data, data.dates[-1], object())
        self.assertEqual({"AAPL": 0.5, "MSFT": 0.5}, weights)

        neutralize, group_for = load_inline_neutralizer(
            "def group_for(symbol, stock):\n"
            "    frame = pd.DataFrame([stock]) if pd is not None else None\n"
            "    return frame.iloc[0]['region'] if frame is not None else stock.get('region', 'Other')\n"
            "\n"
            "def neutralize(alpha, groups):\n"
            "    return dict(alpha)\n"
        )
        self.assertEqual("United States", group_for("AAPL", {"region": "United States"}))
        self.assertEqual({"AAPL": 1.0}, neutralize({"AAPL": 1.0}, {"AAPL": "United States"}))

    def test_old_subindustry_neutralizer_is_migrated_to_sector(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "neutralization_library.json"
            path.write_text(
                '{"neutralizers": [{"id": "none", "name": "None", "code": "def neutralize(alpha, groups):\\n    return dict(alpha)\\n"}, {"id": "subindustry", "name": "Subindustry", "code": "def neutralize(alpha, groups):\\n    return dict(alpha)\\n"}]}',
                encoding="utf-8",
            )
            with patch("portfoliolab.app.NEUTRALIZATION_LIBRARY_PATH", path):
                library = _neutralization_library_payload()

        ids = {item["id"] for item in library["neutralizers"]}
        self.assertIn("region", ids)
        self.assertIn("sector", ids)
        self.assertIn("industry", ids)
        self.assertNotIn("subindustry", ids)

    def test_neutralization_groups_use_current_region_sector_and_industry_metadata(self):
        stocks = [
            {
                "symbol": "AAA",
                "name": "AAA",
                "region": "United States",
                "source": "Yahoo Finance",
                "sector": "Healthcare",
                "industry": "Biotechnology",
            },
            {
                "symbol": "BBB",
                "name": "BBB",
                "region": "Japan",
                "source": "Yahoo Finance",
                "sector": "Technology",
                "industry": "Software",
            },
        ]
        catalog = _stocks_from_payload(stocks)

        self.assertEqual({"AAA": "United States", "BBB": "Japan"}, _neutralization_groups(["AAA", "BBB"], "region", catalog))
        self.assertEqual({"AAA": "Healthcare", "BBB": "Technology"}, _neutralization_groups(["AAA", "BBB"], "sector", catalog))
        self.assertEqual({"AAA": "Biotechnology", "BBB": "Software"}, _neutralization_groups(["AAA", "BBB"], "industry", catalog))

    def test_custom_group_for_code_controls_neutralization_groups(self):
        stocks = [
            {"symbol": "AAA", "name": "AAA", "region": "United States", "source": "Yahoo Finance", "sector": "Healthcare", "industry": "Biotechnology"},
            {"symbol": "BBB", "name": "BBB", "region": "Japan", "source": "Yahoo Finance", "sector": "Technology", "industry": "Software"},
        ]
        catalog = _stocks_from_payload(stocks)

        groups = _neutralization_groups(
            ["AAA", "BBB"],
            "market",
            catalog,
            lambda symbol, stock: stock.get("region") if symbol == "AAA" else "Custom Group",
        )

        self.assertEqual({"AAA": "United States", "BBB": "Custom Group"}, groups)

    def test_market_neutralization_makes_weight_sum_zero(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        pool = next(item for item in catalog["pools"] if item["id"] == "global_quality")
        result = _run_payload_backtest(
            data,
            {
                "symbols": pool["symbols"],
                "weights": {symbol: 1 / len(pool["symbols"]) for symbol in pool["symbols"]},
                "portfolioMode": "manual",
                "signalStrategy": "custom",
                "code": "class Strategy:\n    def compute(self, data, as_of, context):\n        return BASE_WEIGHTS\n",
                "neutralizationMode": "market",
                "neutralizationCode": "def neutralize(alpha, groups):\n    mean_alpha = sum(alpha.values()) / len(alpha)\n    return {symbol: value - mean_alpha for symbol, value in alpha.items()}\n",
                "start": catalog["defaults"]["start"],
                "end": catalog["defaults"]["end"],
                "splitDate": catalog["defaults"]["splitDate"],
                "rebalanceFrequency": "monthly",
            },
        )

        self.assertAlmostEqual(sum(result["finalWeights"].values()), 0.0, places=8)

    def test_market_neutralized_momentum_keeps_nonzero_long_short_book(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        pool = next(item for item in catalog["pools"] if item["id"] == "global_quality")
        result = _run_payload_backtest(
            data,
            {
                "symbols": pool["symbols"],
                "weights": {symbol: 1 / len(pool["symbols"]) for symbol in pool["symbols"]},
                "portfolioMode": "equal",
                "signalStrategy": "momentum",
                "neutralizationMode": "market",
                "neutralizationCode": "def neutralize(alpha, groups):\n    mean_alpha = sum(alpha.values()) / len(alpha)\n    return {symbol: value - mean_alpha for symbol, value in alpha.items()}\n",
                "start": catalog["defaults"]["start"],
                "end": catalog["defaults"]["end"],
                "splitDate": catalog["defaults"]["splitDate"],
                "rebalanceFrequency": "monthly",
            },
        )

        self.assertGreater(sum(abs(weight) for weight in result["finalWeights"].values()), 0)
        self.assertAlmostEqual(sum(result["finalWeights"].values()), 0.0, places=8)

    def test_market_neutralized_mean_reversion_keeps_nonzero_book_for_small_baskets(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        symbols = ["AAPL", "MSFT", "NVDA", "WMT"]
        result = _run_payload_backtest(
            data,
            {
                "symbols": symbols,
                "weights": {symbol: 1 / len(symbols) for symbol in symbols},
                "portfolioMode": "equal",
                "signalStrategy": "reversion",
                "neutralizationMode": "market",
                "neutralizationCode": "def neutralize(alpha, groups):\n    mean_alpha = sum(alpha.values()) / len(alpha)\n    return {symbol: value - mean_alpha for symbol, value in alpha.items()}\n",
                "start": catalog["defaults"]["start"],
                "end": catalog["defaults"]["end"],
                "splitDate": catalog["defaults"]["splitDate"],
                "rebalanceFrequency": "monthly",
            },
        )

        self.assertGreater(sum(abs(weight) for weight in result["finalWeights"].values()), 0)
        self.assertAlmostEqual(sum(result["finalWeights"].values()), 0.0, places=8)

    def test_default_mean_reversion_template_neutralizes_to_nonzero_book(self):
        data = MarketData.demo()
        catalog = _catalog_payload(data)
        symbols = ["AAPL", "MSFT", "NVDA", "WMT"]
        result = _run_payload_backtest(
            data,
            {
                "symbols": symbols,
                "weights": {symbol: 1 / len(symbols) for symbol in symbols},
                "portfolioMode": "equal",
                "signalStrategy": "custom",
                "code": DEFAULT_REVERSION_CODE,
                "neutralizationMode": "market",
                "neutralizationCode": "def neutralize(alpha, groups):\n    mean_alpha = sum(alpha.values()) / len(alpha)\n    return {symbol: value - mean_alpha for symbol, value in alpha.items()}\n",
                "start": catalog["defaults"]["start"],
                "end": catalog["defaults"]["end"],
                "splitDate": catalog["defaults"]["splitDate"],
                "rebalanceFrequency": "monthly",
            },
        )

        self.assertIn("strongest_selected_score", DEFAULT_REVERSION_CODE)
        self.assertGreater(sum(abs(weight) for weight in result["finalWeights"].values()), 0)
        self.assertAlmostEqual(sum(result["finalWeights"].values()), 0.0, places=8)
