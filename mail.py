import os
from typing import Dict, List, Tuple, Optional

import resend
from dotenv import load_dotenv


def send_email_alert(subject: str, body: str):
    """
    Send an email alert using the Resend API.
    
    Args:
        subject: Email subject line.
        body: Email body (HTML supported).
    """
    load_dotenv()
    
    to_email = os.getenv("TO_EMAIL")
    resend_api_key = os.getenv("RESEND_API_KEY")
    from_email = os.getenv("FROM_EMAIL", "onboarding@resend.dev")
    
    if not resend_api_key:
        print("⚠️ RESEND_API_KEY not configured. Email not sent.", flush=True)
        return
    
    resend.api_key = resend_api_key
    
    try:
        resend.Emails.send({
            "from": from_email,
            "to": to_email,
            "subject": subject,
            "html": body
        })
        print('✅ Email sent successfully.', flush=True)
    except Exception as e:
        print(f"❌ Failed to send email: {e}", flush=True)

