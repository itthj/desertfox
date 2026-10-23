"""
Privacy defense and automated takedown notice generator for DesertFox.
Allows individuals to generate formal GDPR (Article 17 Right to Erasure),
CCPA, and DMCA takedown requests for discovered unauthorized photo exposure.
"""

from datetime import datetime, timezone
from typing import Dict, Any

class TakedownNoticeGenerator:
    """Generates legally structured privacy and removal requests."""

    @staticmethod
    def generate_gdpr_erasure_request(
        subject_name: str,
        subject_email: str,
        infringing_url: str,
        image_description: str = "Personal biometric facial photograph",
        additional_notes: str = ""
    ) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notice = f"""SUBJECT: Formal Request for Erasure (GDPR Article 17) - Unauthorized Personal Image
DATE: {date_str}
TO: Data Protection Officer / Site Administrator

Dear Sir or Madam,

I am writing to you in my capacity as a data subject to formally exercise my Right to Erasure (Right to be Forgotten) pursuant to Article 17 of the General Data Protection Regulation (EU) 2016/679 (GDPR).

1. IDENTIFICATION OF PERSONAL DATA:
The specific personal data in question consists of my photograph/biometric image published without my consent:
Target URL: {infringing_url}
Description: {image_description}

2. LEGAL BASIS FOR ERASURE:
Under GDPR Article 17(1)(a) and (b), personal data must be erased without undue delay where:
- The personal data is no longer necessary in relation to the purposes for which it was collected.
- I hereby withdraw any consent on which the processing was based, and there is no other legal ground for the processing.
- The processing of biometric and photographic data concerning identifiable natural persons without explicit consent violates Article 6 and Article 9 of the GDPR.

3. REQUIRED ACTION:
I formally request that you:
a) Immediately delete and permanently erase the specified image and all associated metadata from your servers, content delivery networks (CDNs), and public caches.
b) Notify any third parties or downstream processors to whom you have disclosed this personal data of this erasure request (Article 19).
c) Confirm in writing within one month (pursuant to Article 12(3)) that the erasure has been completed.

{f"Additional Notes: {additional_notes}" if additional_notes else ""}

Sincerely,
{subject_name}
Contact Email: {subject_email}
"""
        return notice.strip()

    @staticmethod
    def generate_dmca_notice(
        copyright_owner: str,
        contact_email: str,
        infringing_url: str,
        original_work_description: str = "Original photographic portrait owned by claimant"
    ) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notice = f"""SUBJECT: Notice of Copyright Infringement pursuant to 17 U.S.C. § 512(c) (DMCA)
DATE: {date_str}
TO: Designated Copyright Agent

I am the copyright owner (or authorized agent) of the original photographic work described below.

1. IDENTIFICATION OF COPYRIGHTED WORK:
Description of Work: {original_work_description}

2. LOCATION OF INFRINGING MATERIAL:
Infringing URL: {infringing_url}

3. STATEMENT OF GOOD FAITH:
I have a good faith belief that use of the material in the manner complained of is not authorized by the copyright owner, its agent, or the law.

4. ACCURACY AND SWORN STATEMENT:
The information in this notification is accurate, and under penalty of perjury, I am the owner of the exclusive right that is allegedly infringed.

Sincerely,
{copyright_owner}
Contact: {contact_email}
"""
        return notice.strip()
