"""Runtime loop: scanning + signals + auto-learning."""

import os
import time

from loguru import logger

# Import runtime modules (will be created in Phase 4)
# from .ftmo_rules import check_daily_loss, check_total_loss
# from .confidence import score_confidence
# from .telegram_bot import send_message

THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
KILL_SWITCH = os.getenv("KILL_SWITCH", "false").lower() == "true"


def loop_once():
    """Execute one iteration of the runtime loop."""
    if KILL_SWITCH:
        logger.warning("Kill-switch is ACTIVE - no signals will be emitted")
        return

    # TODO: Implement runtime loop
    # Dummy state → pretend we generated one HIGH signal
    # confidence = score_confidence(0.72, 0.70, 0.68)
    # if confidence >= THRESHOLD and check_daily_loss(100000, -500) and check_total_loss(100000, 0):
    #     send_message(f"[SMOKE] HIGH signal @ {round(confidence*100,1)}%")
    logger.info("Runtime loop iteration (stub)")


if __name__ == "__main__":
    logger.info("Runtime starting (stub).")
    logger.info(f"Confidence threshold: {THRESHOLD}")
    logger.info(f"Kill-switch: {'ACTIVE' if KILL_SWITCH else 'INACTIVE'}")

    # Short run for smoke testing
    for _ in range(3):
        loop_once()
        time.sleep(1)

    logger.info("Runtime exiting (stub).")
