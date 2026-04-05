import os
from typing import Dict, List, Tuple, Optional

import resend
from dotenv import load_dotenv
import datetime as dt

def get_IST():
    """Get current time in IST timezone."""
    utc_now = dt.datetime.utcnow()
    ist_offset = dt.timedelta(hours=5, minutes=30)
    ist_now = utc_now + ist_offset
    return ist_now.strftime("%Y-%m-%d %H:%M:%S")


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

