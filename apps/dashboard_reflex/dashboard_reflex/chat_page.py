"""
HYDRA Chat Interface Page

Interactive chat with gladiators + recommendations
"""

import reflex as rx
from typing import List, Dict, Any
from datetime import datetime
import json
import sys
from pathlib import Path

# Add libs to path so we can import HydraChat
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "libs"))

from hydra.chat_interface import HydraChat
from hydra.engines.engine_a_deepseek import EngineA_DeepSeek
from hydra.engines.engine_b_claude import EngineB_Claude
from hydra.engines.engine_c_grok import EngineC_Grok
from hydra.engines.engine_d_gemini import EngineD_Gemini


class ChatState(rx.State):
    """State for HYDRA chat interface"""

    # Chat messages
    messages: List[Dict[str, Any]] = []
    user_input: str = ""
    is_loading: bool = False

    # Recommendation state
    last_recommendation: Dict[str, Any] = {}
    feedback_helpful: bool = False
    feedback_accurate: bool = False
    feedback_notes: str = ""

    # Gladiator instances (singleton)
    _gladiators: Dict = {}
    _chat: HydraChat = None

    def initialize_engines(self):
        """Initialize gladiator instances (lazy loading)"""
        if not self._gladiators:
            self._gladiators = {
                "A": EngineA_DeepSeek(),
                "B": EngineB_Claude(),
                "C": EngineC_Grok(),
                "D": EngineD_Gemini(),
            }
            self._chat = HydraChat(self._gladiators)

    async def send_message(self):
        """Send user message to gladiators"""
        if not self.user_input.strip():
            return

        self.is_loading = True

        # Add user message to chat
        user_msg = {
            "sender": "You",
            "text": self.user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "user"
        }
        self.messages.append(user_msg)

        # Initialize gladiators if needed
        self.initialize_engines()

        # Determine target gladiator
        target = "all"
        if "@A" in self.user_input or "@Gladiator-A" in self.user_input:
            target = "A"
        elif "@B" in self.user_input or "@Gladiator-B" in self.user_input:
            target = "B"
        elif "@C" in self.user_input or "@Gladiator-C" in self.user_input:
            target = "C"
        elif "@D" in self.user_input or "@Gladiator-D" in self.user_input:
            target = "D"

        # Get responses from engines
        try:
            responses = self._chat.ask_engine(
                user_message=self.user_input,
                target=target
            )

            # Add gladiator responses
            for engine_name, response_text in responses.items():
                glad_msg = {
                    "sender": f"Gladiator {engine_name}",
                    "text": response_text,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "gladiator",
                    "gladiator": engine_name
                }
                self.messages.append(glad_msg)

        except Exception as e:
            error_msg = {
                "sender": "System",
                "text": f"Error: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "error"
            }
            self.messages.append(error_msg)

        self.user_input = ""
        self.is_loading = False

    async def get_recommendation(self):
        """Get recommendation from all gladiators"""
        self.is_loading = True

        # Initialize gladiators if needed
        self.initialize_engines()

        # For demo, use current market data (in production, fetch real-time)
        try:
            recommendation = self._chat.get_recommendation(
                asset="BTC-USD",
                market_data={
                    "close": 96000,
                    "volume": 1000000,
                    "atr": 1500,
                    "spread": 5
                },
                regime="RANGING",
                ask_all=True
            )

            self.last_recommendation = recommendation

            # Add recommendation message
            rec_msg = {
                "sender": "HYDRA Recommendation",
                "text": self._format_recommendation(recommendation),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "recommendation"
            }
            self.messages.append(rec_msg)

        except Exception as e:
            error_msg = {
                "sender": "System",
                "text": f"Error getting recommendation: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "error"
            }
            self.messages.append(error_msg)

        self.is_loading = False

    def save_feedback(self):
        """Save user feedback on recommendation"""
        if not self.last_recommendation:
            return

        # Initialize chat if needed
        self.initialize_engines()

        feedback = {
            "helpful": self.feedback_helpful,
            "accurate": self.feedback_accurate,
            "notes": self.feedback_notes
        }

        self._chat.save_feedback(
            recommendation_id=self.last_recommendation.get("timestamp", "unknown"),
            feedback=feedback
        )

        # Add confirmation message
        confirm_msg = {
            "sender": "System",
            "text": "Feedback saved! Thank you for helping improve HYDRA.",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "type": "success"
        }
        self.messages.append(confirm_msg)

        # Reset feedback form
        self.feedback_helpful = False
        self.feedback_accurate = False
        self.feedback_notes = ""

    def _format_recommendation(self, rec: Dict) -> str:
        """Format recommendation for display"""
        lines = [
            f"Asset: {rec['asset']} ({rec['regime']} regime)",
            f"Consensus: {rec['consensus']}",
            "",
            "Gladiator Votes:"
        ]

        for glad, vote_data in rec.get("recommendations", {}).items():
            vote = vote_data.get("vote", "UNKNOWN")
            conf = vote_data.get("confidence", 0)
            reasoning = vote_data.get("reasoning", "No reasoning provided")
            lines.append(f"  {glad}: {vote} ({conf:.0%}) - {reasoning}")

        return "\n".join(lines)


def chat_message(message: Dict) -> rx.Component:
    """Render a single chat message"""
    return rx.cond(
        message["type"] == "user",
        # User message
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text(message["sender"], weight="bold", size="2"),
                    rx.text(message["timestamp"], size="1", color="gray"),
                    spacing="2",
                ),
                rx.text(message["text"], size="2", white_space="pre-wrap"),
                align="start",
                spacing="1",
            ),
            background_color="var(--blue-3)",
            padding="3",
            border_radius="8px",
            max_width="80%",
            align_self="end",
        ),
        rx.cond(
            message["type"] == "gladiator",
            # Gladiator message
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text(message["sender"], weight="bold", size="2"),
                        rx.text(message["timestamp"], size="1", color="gray"),
                        spacing="2",
                    ),
                    rx.text(message["text"], size="2", white_space="pre-wrap"),
                    align="start",
                    spacing="1",
                ),
                background_color=rx.cond(
                    message.get("gladiator") == "A", "var(--cyan-3)",
                    rx.cond(message.get("gladiator") == "B", "var(--purple-3)",
                    rx.cond(message.get("gladiator") == "C", "var(--orange-3)",
                    rx.cond(message.get("gladiator") == "D", "var(--green-3)", "var(--gray-3)")
                    ))
                ),
                padding="3",
                border_radius="8px",
                max_width="80%",
                align_self="start",
            ),
            rx.cond(
                message["type"] == "recommendation",
                # Recommendation message
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.text(message["sender"], weight="bold", size="2"),
                            rx.text(message["timestamp"], size="1", color="gray"),
                            spacing="2",
                        ),
                        rx.text(message["text"], size="2", white_space="pre-wrap"),
                        align="start",
                        spacing="1",
                    ),
                    background_color="var(--green-3)",
                    padding="3",
                    border_radius="8px",
                    max_width="80%",
                    align_self="start",
                ),
                # Error or other message types
                rx.box(
                    rx.vstack(
                        rx.hstack(
                            rx.text(message["sender"], weight="bold", size="2"),
                            rx.text(message["timestamp"], size="1", color="gray"),
                            spacing="2",
                        ),
                        rx.text(message["text"], size="2", white_space="pre-wrap"),
                        align="start",
                        spacing="1",
                    ),
                    background_color=rx.cond(message["type"] == "error", "var(--red-3)", "var(--gray-3)"),
                    padding="3",
                    border_radius="8px",
                    max_width="80%",
                    align_self="start",
                ),
            ),
        ),
    )


def chat_interface() -> rx.Component:
    """Main chat interface"""
    return rx.container(
        rx.vstack(
            # Navigation
            rx.hstack(
                rx.link(
                    rx.button("Dashboard", variant="soft", color_scheme="blue"),
                    href="/",
                ),
                rx.link(
                    rx.button("Chat", variant="soft", color_scheme="purple"),
                    href="/chat",
                ),
                spacing="2",
                margin_bottom="4",
            ),

            # Header
            rx.heading("Chat with HYDRA Gladiators", size="7"),
            rx.text(
                "Ask questions, get recommendations, and provide feedback",
                size="3",
                color="gray",
            ),
            rx.divider(),

            # Chat history
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(
                        ChatState.messages,
                        chat_message
                    ),
                    spacing="3",
                    width="100%",
                ),
                height="500px",
                width="100%",
            ),

            # Loading indicator
            rx.cond(
                ChatState.is_loading,
                rx.text("Thinking...", size="2", color="gray"),
            ),

            # Input area
            rx.hstack(
                rx.input(
                    placeholder="Ask anything... (use @A, @B, @C, @D to target specific gladiators)",
                    value=ChatState.user_input,
                    on_change=ChatState.set_user_input,
                    width="100%",
                    size="3",
                ),
                rx.button(
                    "Send",
                    on_click=ChatState.send_message,
                    disabled=ChatState.is_loading,
                    size="3",
                ),
                rx.button(
                    "Get Recommendation",
                    on_click=ChatState.get_recommendation,
                    disabled=ChatState.is_loading,
                    color_scheme="green",
                    size="3",
                ),
                spacing="2",
                width="100%",
            ),

            # Feedback section (only shown after recommendation)
            rx.cond(
                ChatState.last_recommendation != {},
                rx.card(
                    rx.vstack(
                        rx.heading("Provide Feedback", size="5"),
                        rx.text("Help improve HYDRA's recommendations", size="2", color="gray"),
                        rx.divider(),

                        rx.checkbox(
                            "This recommendation was helpful",
                            checked=ChatState.feedback_helpful,
                            on_change=ChatState.set_feedback_helpful,
                        ),
                        rx.checkbox(
                            "This recommendation was accurate",
                            checked=ChatState.feedback_accurate,
                            on_change=ChatState.set_feedback_accurate,
                        ),
                        rx.text_area(
                            placeholder="Additional notes (optional)...",
                            value=ChatState.feedback_notes,
                            on_change=ChatState.set_feedback_notes,
                            width="100%",
                        ),
                        rx.button(
                            "Submit Feedback",
                            on_click=ChatState.save_feedback,
                            color_scheme="blue",
                        ),
                        spacing="3",
                        align="start",
                    ),
                    width="100%",
                    margin_top="4",
                ),
            ),

            # Example questions
            rx.card(
                rx.vstack(
                    rx.heading("Example Questions", size="5"),
                    rx.text("• Why did you vote SELL on the last BTC-USD signal?", size="2"),
                    rx.text("• What's your analysis of ETH-USD right now?", size="2"),
                    rx.text("• @A What structural edges do you see in SOL-USD?", size="2"),
                    rx.text("• @B Is there any logical flaw in the current strategy?", size="2"),
                    rx.text("• @C Have we seen this pattern before in history?", size="2"),
                    rx.text("• @D What's your final recommendation for BTC-USD?", size="2"),
                    spacing="2",
                    align="start",
                ),
                width="100%",
                margin_top="4",
            ),

            spacing="4",
            width="100%",
            padding="4",
        ),
        max_width="1000px",
        on_mount=ChatState.initialize_engines,
    )
