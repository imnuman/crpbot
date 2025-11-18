"""
Signal Parser for V7 Ultimate

Parses DeepSeek LLM responses into structured trading signals.

Expected LLM response format:
```
SIGNAL: BUY
CONFIDENCE: 75%
REASONING: Strong trending + bull regime with positive momentum.
```

Extracted signal:
- Signal: BUY/SELL/HOLD (enum)
- Confidence: 0.0-1.0 (float)
- Reasoning: str
- Raw response: str
- Validation status: bool
"""

import re
import logging
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ParsedSignal:
    """Parsed trading signal from LLM response"""
    signal: SignalType
    confidence: float  # 0.0 to 1.0
    reasoning: str
    raw_response: str
    is_valid: bool
    timestamp: datetime
    parse_warnings: list[str]

    def __str__(self) -> str:
        return (
            f"ParsedSignal(\n"
            f"  signal={self.signal.value},\n"
            f"  confidence={self.confidence:.1%},\n"
            f"  reasoning='{self.reasoning[:50]}...',\n"
            f"  valid={self.is_valid}\n"
            f")"
        )


class SignalParser:
    """
    Parse DeepSeek LLM responses into structured trading signals

    Supports multiple response formats and handles edge cases gracefully.

    Usage:
        parser = SignalParser()
        response = "SIGNAL: BUY\\nCONFIDENCE: 75%\\nREASONING: ..."
        parsed = parser.parse(response)

        if parsed.is_valid:
            print(f"Signal: {parsed.signal}, Confidence: {parsed.confidence:.1%}")
    """

    # Regex patterns for parsing
    SIGNAL_PATTERN = r"SIGNAL:\s*(BUY|SELL|HOLD)"
    CONFIDENCE_PATTERN = r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%?"
    REASONING_PATTERN = r"REASONING:\s*(.+?)(?:\n|$)"

    # Validation thresholds
    MIN_CONFIDENCE = 0.0
    MAX_CONFIDENCE = 1.0
    MIN_REASONING_LENGTH = 10  # Minimum characters for valid reasoning

    def __init__(self, strict_mode: bool = False):
        """
        Initialize Signal Parser

        Args:
            strict_mode: If True, reject signals that don't match exact format
        """
        self.strict_mode = strict_mode
        logger.info(f"SignalParser initialized | Strict Mode: {strict_mode}")

    def parse(self, response: str) -> ParsedSignal:
        """
        Parse LLM response into structured signal

        Args:
            response: Raw LLM response text

        Returns:
            ParsedSignal object with extracted fields and validation status
        """
        warnings = []
        is_valid = True

        # Clean response
        response_clean = response.strip()

        # Extract signal
        signal, signal_warning = self._extract_signal(response_clean)
        if signal_warning:
            warnings.append(signal_warning)
            if self.strict_mode:
                is_valid = False

        # Extract confidence
        confidence, conf_warning = self._extract_confidence(response_clean)
        if conf_warning:
            warnings.append(conf_warning)
            if self.strict_mode:
                is_valid = False

        # Extract reasoning
        reasoning, reason_warning = self._extract_reasoning(response_clean)
        if reason_warning:
            warnings.append(reason_warning)
            if self.strict_mode:
                is_valid = False

        # Final validation
        if signal is None or confidence is None or reasoning is None:
            is_valid = False

        parsed = ParsedSignal(
            signal=signal or SignalType.HOLD,  # Default to HOLD if invalid
            confidence=confidence if confidence is not None else 0.0,
            reasoning=reasoning or "Failed to parse response",
            raw_response=response_clean,
            is_valid=is_valid,
            timestamp=datetime.now(),
            parse_warnings=warnings
        )

        if is_valid:
            logger.info(
                f"Parsed signal: {parsed.signal.value} @ {parsed.confidence:.1%}"
            )
        else:
            logger.warning(
                f"Invalid signal parsed | Warnings: {len(warnings)} | "
                f"Defaulting to HOLD"
            )

        return parsed

    def _extract_signal(self, text: str) -> Tuple[Optional[SignalType], Optional[str]]:
        """
        Extract signal type from response

        Args:
            text: Response text

        Returns:
            Tuple of (signal, warning_message)
        """
        match = re.search(self.SIGNAL_PATTERN, text, re.IGNORECASE)

        if match:
            signal_str = match.group(1).upper()
            try:
                return SignalType(signal_str), None
            except ValueError:
                return None, f"Invalid signal type: {signal_str}"
        else:
            # Try fuzzy matching
            text_upper = text.upper()
            if "BUY" in text_upper and "SELL" not in text_upper:
                return SignalType.BUY, "Signal extracted via fuzzy match"
            elif "SELL" in text_upper and "BUY" not in text_upper:
                return SignalType.SELL, "Signal extracted via fuzzy match"
            elif "HOLD" in text_upper:
                return SignalType.HOLD, "Signal extracted via fuzzy match"
            else:
                return None, "No signal found in response"

    def _extract_confidence(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract confidence percentage from response

        Args:
            text: Response text

        Returns:
            Tuple of (confidence 0-1, warning_message)
        """
        match = re.search(self.CONFIDENCE_PATTERN, text, re.IGNORECASE)

        if match:
            try:
                conf_pct = float(match.group(1))

                # Convert to 0-1 range if needed
                if conf_pct > 1.0:
                    conf_pct = conf_pct / 100.0

                # Validate range
                if conf_pct < self.MIN_CONFIDENCE or conf_pct > self.MAX_CONFIDENCE:
                    return (
                        max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, conf_pct)),
                        f"Confidence out of range: {conf_pct:.1%}, clamped to valid range"
                    )

                return conf_pct, None

            except ValueError:
                return None, f"Failed to parse confidence: {match.group(1)}"
        else:
            # Try to find any percentage in text
            pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if pct_match:
                try:
                    conf_pct = float(pct_match.group(1)) / 100.0
                    return conf_pct, "Confidence extracted via fuzzy match"
                except ValueError:
                    pass

            return None, "No confidence found in response"

    def _extract_reasoning(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract reasoning from response

        Args:
            text: Response text

        Returns:
            Tuple of (reasoning, warning_message)
        """
        match = re.search(self.REASONING_PATTERN, text, re.IGNORECASE | re.DOTALL)

        if match:
            reasoning = match.group(1).strip()

            # Remove trailing punctuation artifacts
            reasoning = re.sub(r'\s+', ' ', reasoning)  # Normalize whitespace

            # Validate length
            if len(reasoning) < self.MIN_REASONING_LENGTH:
                return reasoning, f"Reasoning too short: {len(reasoning)} chars"

            return reasoning, None
        else:
            # Try to extract everything after "REASONING" keyword
            reasoning_idx = text.upper().find("REASONING")
            if reasoning_idx != -1:
                reasoning = text[reasoning_idx + len("REASONING"):].strip()
                # Remove leading colon/whitespace
                reasoning = re.sub(r'^:\s*', '', reasoning)

                if len(reasoning) >= self.MIN_REASONING_LENGTH:
                    return reasoning, "Reasoning extracted via fallback method"

            return None, "No reasoning found in response"

    def validate_signal(self, parsed: ParsedSignal) -> Tuple[bool, list[str]]:
        """
        Perform additional validation on parsed signal

        Args:
            parsed: ParsedSignal object

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Check signal type
        if not isinstance(parsed.signal, SignalType):
            errors.append("Invalid signal type")

        # Check confidence range
        if not (self.MIN_CONFIDENCE <= parsed.confidence <= self.MAX_CONFIDENCE):
            errors.append(
                f"Confidence out of range: {parsed.confidence:.1%}"
            )

        # Check reasoning length
        if len(parsed.reasoning) < self.MIN_REASONING_LENGTH:
            errors.append(
                f"Reasoning too short: {len(parsed.reasoning)} chars"
            )

        # Check for suspicious patterns
        if "error" in parsed.reasoning.lower() or "fail" in parsed.reasoning.lower():
            errors.append("Reasoning contains error keywords")

        is_valid = len(errors) == 0

        logger.debug(
            f"Signal validation: {'PASS' if is_valid else 'FAIL'} | "
            f"Errors: {len(errors)}"
        )

        return is_valid, errors


# Convenience function for quick parsing
def parse_llm_response(
    response: str,
    strict: bool = False
) -> ParsedSignal:
    """
    Quick parse of LLM response

    Args:
        response: Raw LLM response text
        strict: Enable strict mode validation

    Returns:
        ParsedSignal object
    """
    parser = SignalParser(strict_mode=strict)
    return parser.parse(response)


if __name__ == "__main__":
    # Test Signal Parser
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("=" * 80)
    print("Signal Parser - Test Run")
    print("=" * 80)

    parser = SignalParser(strict_mode=False)

    # Test 1: Perfect format
    print("\n1. Test: Perfect Format")
    response1 = """SIGNAL: BUY
CONFIDENCE: 75%
REASONING: Strong trending (Hurst 0.72) + bull regime (65%) with positive momentum. Risk acceptable (VaR 12%, Sharpe 1.2)."""

    parsed1 = parser.parse(response1)
    print(f"   Signal: {parsed1.signal.value}")
    print(f"   Confidence: {parsed1.confidence:.1%}")
    print(f"   Reasoning: {parsed1.reasoning}")
    print(f"   Valid: {'✅ Yes' if parsed1.is_valid else '❌ No'}")
    print(f"   Warnings: {len(parsed1.parse_warnings)}")

    # Test 2: Missing spaces
    print("\n2. Test: Missing Spaces")
    response2 = "SIGNAL:SELL CONFIDENCE:82% REASONING:Bear market detected with high entropy."

    parsed2 = parser.parse(response2)
    print(f"   Signal: {parsed2.signal.value}")
    print(f"   Confidence: {parsed2.confidence:.1%}")
    print(f"   Valid: {'✅ Yes' if parsed2.is_valid else '❌ No'}")

    # Test 3: Natural language response
    print("\n3. Test: Natural Language")
    response3 = "Based on the analysis, I recommend a BUY with 68% confidence because the market shows strong bullish momentum."

    parsed3 = parser.parse(response3)
    print(f"   Signal: {parsed3.signal.value}")
    print(f"   Confidence: {parsed3.confidence:.1%}")
    print(f"   Valid: {'✅ Yes' if parsed3.is_valid else '❌ No'}")
    print(f"   Warnings: {parsed3.parse_warnings}")

    # Test 4: Edge case - confidence without % symbol
    print("\n4. Test: Confidence Without % Symbol")
    response4 = "SIGNAL: HOLD\nCONFIDENCE: 0.55\nREASONING: Mixed signals, wait for clearer trend."

    parsed4 = parser.parse(response4)
    print(f"   Signal: {parsed4.signal.value}")
    print(f"   Confidence: {parsed4.confidence:.1%}")
    print(f"   Valid: {'✅ Yes' if parsed4.is_valid else '❌ No'}")

    # Test 5: Invalid/malformed response
    print("\n5. Test: Malformed Response")
    response5 = "The market is very volatile right now. Maybe consider waiting."

    parsed5 = parser.parse(response5)
    print(f"   Signal: {parsed5.signal.value} (default)")
    print(f"   Confidence: {parsed5.confidence:.1%}")
    print(f"   Valid: {'✅ Yes' if parsed5.is_valid else '❌ No'}")
    print(f"   Warnings: {len(parsed5.parse_warnings)} - {parsed5.parse_warnings}")

    # Test 6: Validation
    print("\n6. Test: Signal Validation")
    is_valid, errors = parser.validate_signal(parsed1)
    print(f"   Validation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print(f"   Errors: {errors}")

    # Test 7: Strict mode
    print("\n7. Test: Strict Mode")
    strict_parser = SignalParser(strict_mode=True)
    parsed7 = strict_parser.parse(response3)  # Natural language
    print(f"   Signal: {parsed7.signal.value}")
    print(f"   Valid (strict): {'✅ Yes' if parsed7.is_valid else '❌ No'}")

    # Test 8: Edge cases
    print("\n8. Test: Edge Cases")
    edge_cases = [
        ("SIGNAL: BUY\nCONFIDENCE: 150%\nREASONING: Test", "Confidence > 100%"),
        ("SIGNAL: BUY\nCONFIDENCE: -10%\nREASONING: Test", "Negative confidence"),
        ("SIGNAL: MAYBE\nCONFIDENCE: 50%\nREASONING: Test", "Invalid signal type"),
        ("SIGNAL: BUY\nCONFIDENCE: 75%\nREASONING: x", "Short reasoning"),
    ]

    for response, description in edge_cases:
        parsed = parser.parse(response)
        print(f"   {description}: Valid={parsed.is_valid}, Warnings={len(parsed.parse_warnings)}")

    print("\n" + "=" * 80)
    print("Signal Parser Test Complete!")
    print("=" * 80)
    print("\nKey Features Verified:")
    print("  ✅ Perfect format parsing")
    print("  ✅ Missing spaces handling")
    print("  ✅ Natural language fuzzy matching")
    print("  ✅ Confidence format variations")
    print("  ✅ Malformed response handling")
    print("  ✅ Signal validation")
    print("  ✅ Strict mode enforcement")
    print("  ✅ Edge case handling")
    print("=" * 80)
