import logging
from datetime import datetime
from ai.intent_recognizer import analyze_intent
from ai.threat_prediction_ai import predict_threat
from ai.auto_feedback_loop import feedback_evaluator

logger = logging.getLogger("PolicyEnforcer")

def enforce_zero_trust_policy(user_profile, system_context):
    """
    Applies adaptive zero-trust enforcement based on behavioral and contextual AI.
    """
    threat_level = predict_threat(user_profile, system_context)
    intent_score = analyze_intent(user_profile.get("recent_activity", ""))
    feedback_risk = feedback_evaluator(user_profile)

    logger.info(f"[ZeroTrust] Threat Level: {threat_level}, Intent Score: {intent_score}, Feedback Risk: {feedback_risk}")

    if threat_level > 0.7 or feedback_risk > 0.6 or intent_score < 0.3:
        logger.warning(f"[ZeroTrust] Access denied for user: {user_profile['id']}")
        return False
    return True

def enforce_stake_limits(stake, symbol):
    from engine.risk_guardian import get_current_risk_level  # moved import here
    """
    Dynamically enforce stake limits based on AI risk scores and trading integrity checks.
    """
    trade_amount = stake
    risk_level = get_current_risk_level()
    intent_score = analyze_intent(symbol)
    
    logger.info(f"[StakeEnforcer] Requested: {trade_amount}, Risk Level: {risk_level}, Intent Score: {intent_score}")

    # Dynamic stake ceiling
    max_allowed = 0.05 * current_balance * (1.0 - risk_level)

    if trade_amount > max_allowed or intent_score < 0.4:
        logger.warning(f"[StakeEnforcer] Trade denied: exceeds safe stake limit or low strategy intent.")
        return False
    return True
