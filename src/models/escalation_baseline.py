import re


# Intents that should normally be reviewed by a human
HIGH_RISK_INTENTS = {
    "Account / Security",
    "Purchase / Billing / Store",
    "Hardware / Accessories",
}


# Keywords that indicate potentially serious situations
ESCALATION_KEYWORDS = [
    "exploded",
    "swollen",
    "smoke",
    "fire",
    "burning",
    "overheating",
    "dangerous",
    "injury",
    "hurt",
    "stolen",
    "hacked",
    "fraud",
    "unauthorized charge",
]


def should_escalate(
    customer_message,
    predicted_intent,
    top_similarity,
):
    """
    Decide whether a customer message should be handled
    automatically or escalated to a human.

    Returns:
        decision: AUTO-HANDLE or ESCALATE
        reason: explanation for the decision
    """

    text = customer_message.lower()

    # -----------------------------------------------
    # Rule 1: Safety / serious incident
    # -----------------------------------------------

    for keyword in ESCALATION_KEYWORDS:
        if keyword in text:
            return (
                "ESCALATE",
                f"Potential safety or serious incident: '{keyword}'"
            )

    # -----------------------------------------------
    # Rule 2: High-risk intents
    # -----------------------------------------------

    if predicted_intent in HIGH_RISK_INTENTS:
        return (
            "ESCALATE",
            f"High-risk intent: {predicted_intent}"
        )

    # -----------------------------------------------
    # Rule 3: Low retrieval confidence
    # -----------------------------------------------

    if top_similarity < 0.55:
        return (
            "ESCALATE",
            f"Low retrieval confidence: {top_similarity:.4f}"
        )

    # -----------------------------------------------
    # Otherwise handle automatically
    # -----------------------------------------------

    return (
        "AUTO-HANDLE",
        "Low-risk issue with sufficient retrieval confidence"
    )


# --------------------------------------------------
# Test examples
# --------------------------------------------------
# --------------------------------------------------
# Test examples
# --------------------------------------------------

if __name__ == "__main__":

    examples = [
        {
            "message": "My iPhone battery is draining very quickly.",
            "intent": "Battery",
            "similarity": 0.8865,
        },
        {
            "message": "Every time I type I get a question mark box.",
            "intent": "Keyboard / Input",
            "similarity": 0.9520,
        },
        {
            "message": "Someone charged my card without my permission.",
            "intent": "Purchase / Billing / Store",
            "similarity": 0.80,
        },
        {
            "message": "My MacBook battery exploded and started smoking.",
            "intent": "Hardware / Accessories",
            "similarity": 0.90,
        },
        {
            "message": "My phone has a strange problem.",
            "intent": "Other / Unclear",
            "similarity": 0.30,
        },
    ]

    print("\n" + "=" * 80)
    print("ESCALATION BASELINE")
    print("=" * 80)

    for example in examples:

        decision, reason = should_escalate(
            example["message"],
            example["intent"],
            example["similarity"],
        )

        print("\nCustomer:", example["message"])
        print("Intent:", example["intent"])
        print("Similarity:", example["similarity"])
        print("Decision:", decision)
        print("Reason:", reason)