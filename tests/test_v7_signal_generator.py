"""
Unit Tests for V7 Ultimate Signal Generator

Tests the complete signal generation pipeline:
- Mathematical analysis (6 theories)
- LLM prompt synthesis
- Response parsing
- Integration orchestration
- Error handling
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import V7 components
from libs.llm import (
    SignalGenerator,
    SignalGenerationResult,
    DeepSeekClient,
    DeepSeekResponse,
    SignalSynthesizer,
    SignalParser,
    ParsedSignal,
    SignalType,
    MarketContext,
    TheoryAnalysis,
)


class TestSignalGeneratorUnit:
    """Unit tests for SignalGenerator (no API calls)"""

    def test_initialization(self):
        """Test SignalGenerator initializes correctly"""
        generator = SignalGenerator(
            api_key="test-key",
            conservative_mode=True,
            strict_parsing=False,
            lookback_window=200
        )

        assert generator.lookback_window == 200
        assert generator.temperature == 0.7
        assert generator.deepseek_client is not None
        assert generator.signal_synthesizer is not None
        assert generator.signal_parser is not None

    def test_initialization_with_env_var(self):
        """Test SignalGenerator uses DEEPSEEK_API_KEY env var"""
        with patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'env-test-key'}):
            generator = SignalGenerator()
            assert generator.deepseek_client.api_key == 'env-test-key'

    def test_mathematical_analysis(self):
        """Test mathematical analysis runs all 6 theories"""
        generator = SignalGenerator(api_key="test-key")

        # Generate synthetic price data
        np.random.seed(42)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        current_price = float(prices[-1])

        # Run mathematical analysis
        analysis = generator._run_mathematical_analysis(prices, current_price)

        # Verify all theory outputs are present
        assert isinstance(analysis, TheoryAnalysis)
        assert 0.0 <= analysis.entropy <= 1.0
        assert 0.0 <= analysis.hurst <= 1.0
        assert analysis.current_regime is not None
        assert isinstance(analysis.regime_probabilities, dict)
        assert analysis.denoised_price > 0
        assert isinstance(analysis.price_momentum, float)
        assert 0.0 <= analysis.win_rate_estimate <= 1.0
        assert isinstance(analysis.risk_metrics, dict)

    def test_mathematical_analysis_entropy_range(self):
        """Test Shannon Entropy output is in valid range"""
        generator = SignalGenerator(api_key="test-key")
        prices = np.linspace(45000, 46000, 250)  # Linear trend

        analysis = generator._run_mathematical_analysis(prices, 46000.0)

        assert 0.0 <= analysis.entropy <= 1.0
        assert 'predictability' in analysis.entropy_interpretation

    def test_mathematical_analysis_hurst_interpretation(self):
        """Test Hurst Exponent interpretation"""
        generator = SignalGenerator(api_key="test-key")

        # Trending market (random walk with drift)
        np.random.seed(42)
        trending_prices = 45000 + np.cumsum(np.random.randn(250) * 50 + 10)

        analysis = generator._run_mathematical_analysis(trending_prices, trending_prices[-1])

        assert isinstance(analysis.hurst_interpretation, str)
        assert analysis.hurst is not None

    def test_bayesian_learning_update(self):
        """Test Bayesian win rate learning updates correctly"""
        generator = SignalGenerator(api_key="test-key")

        initial_win_rate = generator.bayesian_learner.get_current_estimate().mean

        # Update with winning trade
        generator.update_bayesian_learning(trade_won=True)

        new_win_rate = generator.bayesian_learner.get_current_estimate().mean

        # Win rate should increase after winning trade
        assert new_win_rate >= initial_win_rate

    def test_statistics_retrieval(self):
        """Test generator statistics retrieval"""
        generator = SignalGenerator(api_key="test-key")

        stats = generator.get_statistics()

        assert 'deepseek_api' in stats
        assert 'bayesian_win_rate' in stats
        assert 'bayesian_confidence_interval' in stats
        assert 'bayesian_total_trades' in stats

    def test_generate_signal_insufficient_data(self):
        """Test error handling with insufficient data"""
        generator = SignalGenerator(api_key="test-key", lookback_window=200)

        # Provide insufficient data
        prices = np.array([45000, 45100, 45200])  # Only 3 points
        timestamps = np.array([1, 2, 3])

        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=45200.0
        )

        # Should fail gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "Insufficient data" in result.error_message
        assert result.parsed_signal.signal == SignalType.HOLD
        assert result.parsed_signal.confidence == 0.0


class TestSignalGeneratorIntegration:
    """Integration tests for SignalGenerator (mocked API calls)"""

    @patch.object(DeepSeekClient, 'chat')
    def test_generate_signal_with_mocked_llm(self, mock_chat):
        """Test complete signal generation with mocked LLM response"""
        # Mock LLM response
        mock_response = DeepSeekResponse(
            content="SIGNAL: BUY\nCONFIDENCE: 75%\nREASONING: Strong trending market with positive momentum indicators.",
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=50,
            total_tokens=650,
            cost_usd=0.00027,
            finish_reason="stop",
            timestamp=datetime.now()
        )
        mock_chat.return_value = mock_response

        # Initialize generator
        generator = SignalGenerator(api_key="test-key")

        # Generate synthetic data
        np.random.seed(42)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        timestamps = np.arange(250) * 3600

        # Generate signal
        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=float(prices[-1]),
            timeframe="1h",
            volume=1_500_000,
            spread=0.02
        )

        # Verify result
        assert result.success is True
        assert result.error_message is None
        assert result.parsed_signal.signal == SignalType.BUY
        assert result.parsed_signal.confidence == 0.75
        assert result.parsed_signal.is_valid is True
        assert result.theory_analysis is not None
        assert result.llm_response is not None
        assert result.total_cost_usd == 0.00027
        assert result.generation_time_seconds > 0

        # Verify LLM was called
        mock_chat.assert_called_once()

    @patch.object(DeepSeekClient, 'chat')
    def test_generate_signal_sell_signal(self, mock_chat):
        """Test SELL signal generation"""
        mock_response = DeepSeekResponse(
            content="SIGNAL: SELL\nCONFIDENCE: 82%\nREASONING: Mean-reversion signals with high entropy indicate selling opportunity.",
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=45,
            total_tokens=645,
            cost_usd=0.00026,
            finish_reason="stop",
            timestamp=datetime.now()
        )
        mock_chat.return_value = mock_response

        generator = SignalGenerator(api_key="test-key")

        np.random.seed(43)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        timestamps = np.arange(250) * 3600

        result = generator.generate_signal(
            symbol="ETH-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=float(prices[-1])
        )

        assert result.success is True
        assert result.parsed_signal.signal == SignalType.SELL
        assert result.parsed_signal.confidence == 0.82

    @patch.object(DeepSeekClient, 'chat')
    def test_generate_signal_hold_signal(self, mock_chat):
        """Test HOLD signal generation"""
        mock_response = DeepSeekResponse(
            content="SIGNAL: HOLD\nCONFIDENCE: 55%\nREASONING: Mixed signals, unclear market direction.",
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=40,
            total_tokens=640,
            cost_usd=0.00025,
            finish_reason="stop",
            timestamp=datetime.now()
        )
        mock_chat.return_value = mock_response

        generator = SignalGenerator(api_key="test-key")

        np.random.seed(44)
        prices = np.ones(250) * 45000  # Flat prices
        timestamps = np.arange(250) * 3600

        result = generator.generate_signal(
            symbol="SOL-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=45000.0
        )

        assert result.success is True
        assert result.parsed_signal.signal == SignalType.HOLD
        assert result.parsed_signal.confidence == 0.55

    @patch.object(DeepSeekClient, 'chat')
    def test_generate_signal_malformed_llm_response(self, mock_chat):
        """Test handling of malformed LLM response"""
        # Return malformed response
        mock_response = DeepSeekResponse(
            content="The market looks interesting today...",  # No structured format
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=30,
            total_tokens=630,
            cost_usd=0.00024,
            finish_reason="stop",
            timestamp=datetime.now()
        )
        mock_chat.return_value = mock_response

        generator = SignalGenerator(api_key="test-key", strict_parsing=False)

        np.random.seed(45)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        timestamps = np.arange(250) * 3600

        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=float(prices[-1])
        )

        # Should succeed but with invalid signal (defaults to HOLD)
        assert result.success is True
        assert result.parsed_signal.is_valid is False
        assert result.parsed_signal.signal == SignalType.HOLD
        assert result.parsed_signal.confidence == 0.0
        assert len(result.parsed_signal.parse_warnings) > 0

    @patch.object(DeepSeekClient, 'chat')
    def test_generate_signal_api_error(self, mock_chat):
        """Test handling of API errors"""
        # Simulate API error
        mock_chat.side_effect = Exception("API connection failed")

        generator = SignalGenerator(api_key="test-key")

        np.random.seed(46)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        timestamps = np.arange(250) * 3600

        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=float(prices[-1])
        )

        # Should fail gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "API connection failed" in result.error_message
        assert result.parsed_signal.signal == SignalType.HOLD
        assert result.parsed_signal.confidence == 0.0
        assert result.total_cost_usd == 0.0


class TestSignalGeneratorCostTracking:
    """Tests for cost tracking and budget management"""

    @patch.object(DeepSeekClient, 'chat')
    def test_cost_tracking_single_signal(self, mock_chat):
        """Test cost tracking for single signal"""
        mock_response = DeepSeekResponse(
            content="SIGNAL: BUY\nCONFIDENCE: 75%\nREASONING: Test",
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=50,
            total_tokens=650,
            cost_usd=0.000217,  # 600*0.27/1M + 50*1.10/1M
            finish_reason="stop",
            timestamp=datetime.now()
        )

        generator = SignalGenerator(api_key="test-key")

        # Create side effect that tracks cost when mock is called
        def mock_chat_with_tracking(*args, **kwargs):
            # Update the client's cost tracking
            generator.deepseek_client.total_cost += mock_response.cost_usd
            generator.deepseek_client.total_tokens += mock_response.total_tokens
            return mock_response

        mock_chat.side_effect = mock_chat_with_tracking

        np.random.seed(47)
        prices = 45000 + np.cumsum(np.random.randn(250) * 100)
        timestamps = np.arange(250) * 3600

        result = generator.generate_signal(
            symbol="BTC-USD",
            prices=prices,
            timestamps=timestamps,
            current_price=float(prices[-1])
        )

        # Verify cost tracking
        assert result.total_cost_usd == 0.000217
        assert result.llm_response.total_tokens == 650

        # Verify generator statistics
        stats = generator.get_statistics()
        assert stats['deepseek_api']['total_cost_usd'] == 0.000217

    @patch.object(DeepSeekClient, 'chat')
    def test_cost_projection_100_signals_per_day(self, mock_chat):
        """Test monthly cost projection for 100 signals/day"""
        mock_response = DeepSeekResponse(
            content="SIGNAL: BUY\nCONFIDENCE: 75%\nREASONING: Test signal.",
            model="deepseek-chat",
            prompt_tokens=600,
            completion_tokens=50,
            total_tokens=650,
            cost_usd=0.000217,
            finish_reason="stop",
            timestamp=datetime.now()
        )
        mock_chat.return_value = mock_response

        generator = SignalGenerator(api_key="test-key")

        # Simulate 10 signals
        np.random.seed(48)
        for i in range(10):
            prices = 45000 + np.cumsum(np.random.randn(250) * 100)
            timestamps = np.arange(250) * 3600

            generator.generate_signal(
                symbol="BTC-USD",
                prices=prices,
                timestamps=timestamps,
                current_price=float(prices[-1])
            )

        stats = generator.get_statistics()
        avg_cost_per_signal = stats['deepseek_api']['avg_cost_per_request']

        # Project monthly cost
        monthly_cost = avg_cost_per_signal * 100 * 30

        # Verify within budget ($100/month)
        assert monthly_cost < 100.0, f"Monthly cost ${monthly_cost:.2f} exceeds $100 budget"


class TestSignalGeneratorEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_exact_lookback_window_size(self):
        """Test with exactly lookback_window data points"""
        generator = SignalGenerator(api_key="test-key", lookback_window=200)

        prices = np.linspace(45000, 46000, 200)  # Exactly 200 points
        timestamps = np.arange(200) * 3600

        with patch.object(DeepSeekClient, 'chat') as mock_chat:
            mock_chat.return_value = DeepSeekResponse(
                content="SIGNAL: BUY\nCONFIDENCE: 75%\nREASONING: Test",
                model="deepseek-chat",
                prompt_tokens=600,
                completion_tokens=50,
                total_tokens=650,
                cost_usd=0.000217,
                finish_reason="stop",
                timestamp=datetime.now()
            )

            result = generator.generate_signal(
                symbol="BTC-USD",
                prices=prices,
                timestamps=timestamps,
                current_price=46000.0
            )

            assert result.success is True

    def test_extreme_price_values(self):
        """Test with extreme price values"""
        generator = SignalGenerator(api_key="test-key")

        # Very high prices
        prices = np.ones(250) * 1_000_000
        timestamps = np.arange(250) * 3600

        with patch.object(DeepSeekClient, 'chat') as mock_chat:
            mock_chat.return_value = DeepSeekResponse(
                content="SIGNAL: HOLD\nCONFIDENCE: 50%\nREASONING: Extreme prices",
                model="deepseek-chat",
                prompt_tokens=600,
                completion_tokens=40,
                total_tokens=640,
                cost_usd=0.000206,
                finish_reason="stop",
                timestamp=datetime.now()
            )

            result = generator.generate_signal(
                symbol="BTC-USD",
                prices=prices,
                timestamps=timestamps,
                current_price=1_000_000.0
            )

            assert result.success is True

    def test_volatile_price_swings(self):
        """Test with highly volatile price data"""
        generator = SignalGenerator(api_key="test-key")

        # Simulate extreme volatility
        np.random.seed(49)
        prices = 45000 + np.cumsum(np.random.randn(250) * 1000)  # High volatility
        timestamps = np.arange(250) * 3600

        with patch.object(DeepSeekClient, 'chat') as mock_chat:
            mock_chat.return_value = DeepSeekResponse(
                content="SIGNAL: HOLD\nCONFIDENCE: 40%\nREASONING: High volatility, unclear direction",
                model="deepseek-chat",
                prompt_tokens=600,
                completion_tokens=45,
                total_tokens=645,
                cost_usd=0.000211,
                finish_reason="stop",
                timestamp=datetime.now()
            )

            result = generator.generate_signal(
                symbol="BTC-USD",
                prices=prices,
                timestamps=timestamps,
                current_price=float(prices[-1])
            )

            assert result.success is True
            # High volatility should be reflected in analysis
            assert result.theory_analysis.entropy > 0.3  # High entropy expected


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
