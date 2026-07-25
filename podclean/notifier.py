"""Module for sending email notifications via Resend."""

import resend

from .config import get_config


def send_notification(subject: str, body: str) -> bool:
    """Send an email notification with the given subject and body.
    
    Returns True if successful, False otherwise.
    """
    config = get_config()
    
    if not config.resend_api_key or not config.resend_from_email or not config.email_recipient:
        print("Resend credentials not fully configured. Skipping notification.")
        return False

    resend.api_key = config.resend_api_key

    try:
        print(f"Sending notification to {config.email_recipient} via Resend...")
        resend.Emails.send({
            "from": config.resend_from_email,
            "to": config.email_recipient,
            "subject": subject,
            "text": body
        })
        print("Notification sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send notification: {e}")
        return False
