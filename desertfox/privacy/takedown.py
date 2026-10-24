"""
Privacy defense and automated takedown notice generator for DesertFox.
Allows individuals to generate formal GDPR (Article 17 Right to Erasure),
CCPA (California Consumer Privacy Act), and DMCA takedown requests
for discovered unauthorized photo exposure.
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

    @staticmethod
    def generate_ccpa_deletion_request(
        consumer_name: str,
        consumer_email: str,
        infringing_url: str,
        consumer_state: str = "California",
        additional_notes: str = ""
    ) -> str:
        """
        Generates a California Consumer Privacy Act (CCPA) §1798.105
        right-to-delete request for unauthorized biometric / photographic data.
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notice = f"""SUBJECT: Verified Consumer Request to Delete Personal Information (CCPA §1798.105)
DATE: {date_str}
TO: Privacy Officer / Data Protection Contact

Dear Sir or Madam,

I am a {consumer_state} resident exercising my right to deletion of personal information under the California Consumer Privacy Act of 2018 (Cal. Civ. Code §1798.100 et seq., as amended by the CPRA).

1. IDENTIFICATION OF PERSONAL INFORMATION:
My biometric and photographic personal information is published without my authorization at:
Target URL: {infringing_url}

2. LEGAL BASIS FOR DELETION:
Under CCPA §1798.105(a), a consumer has the right to request that a business delete any personal information about the consumer which the business has collected. Biometric information, including facial imagery and derived identifiers, constitutes "personal information" under §1798.140(v)(1)(E).

Upon receiving a verifiable consumer request, the business must delete the consumer's personal information from its records and direct any service providers or contractors to delete the consumer's personal information from their records (§1798.105(c)).

3. EXCEPTIONS REVIEW:
I have reviewed the exceptions listed in §1798.105(d) and believe none apply to this data. The images in question were not collected for a transaction, are not necessary to detect security incidents, and are not required for internal uses reasonably aligned with my expectations as a consumer.

4. VERIFICATION:
I am prepared to verify my identity as the consumer whose data is the subject of this request, in accordance with §1798.140(ak) and the regulations adopted pursuant to §1798.185.

5. REQUIRED ACTION:
I formally request that you:
a) Delete all copies of the specified biometric/photographic data and associated personal information within 45 days (§1798.105(b)).
b) Notify all service providers, contractors, and third parties to whom you disclosed this data to also delete it (§1798.105(c)).
c) Confirm completion of deletion in writing.

6. NON-DISCRIMINATION:
Pursuant to §1798.125, I expect that exercising this right will not result in any discriminatory treatment, including denial of goods or services, different pricing, or different quality.

{f"Additional Notes: {additional_notes}" if additional_notes else ""}

Sincerely,
{consumer_name}
Contact Email: {consumer_email}
Resident of: {consumer_state}
"""
        return notice.strip()

    @staticmethod
    def generate_ncii_emergency_takedown(
        subject_name: str,
        subject_email: str,
        infringing_url: str,
        additional_notes: str = "",
    ) -> str:
        """
        Generates an Emergency Non-Consensual Intimate Imagery (NCII) / Leaked Media
        & Video Removal Demand citing the U.S. TAKE IT DOWN Act, 18 U.S.C. § 2257,
        17 U.S.C. § 512(c) (DMCA), EU Digital Services Act (DSA) Art. 16, and GDPR Art. 17/9.
        """
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notice = f"""SUBJECT: URGENT LEGAL NOTICE — Non-Consensual Intimate Media / Leaked Content Removal Demand (TAKE IT DOWN Act / 18 U.S.C. § 2257 / DMCA § 512(c) / DSA Art. 16)
DATE: {date_str}
TO: Trust & Safety / Abuse Desk / Designated Copyright Agent / Legal Compliance

IMMEDIATE EXPEDITED TAKEDOWN REQUIRED (48-HOUR STATUTORY MANDATE)

I am the identifiable individual depicted in private, unauthorized, and non-consensual photographic and/or video media currently hosted or indexed on your platform at the following URL(s):

Infringing & Unauthorized URL: {infringing_url}

1. NON-CONSENSUAL INTIMATE IMAGERY (NCII) & STATUTORY VIOLATIONS:
The material at the above URL was published and distributed without my consent. Continued hosting of this content violates:
- U.S. Federal TAKE IT DOWN Act & State NCII Statutes (mandating removal of non-consensual intimate visual depictions and associated copies within 48 hours of notice).
- 18 U.S.C. § 2257 & 28 C.F.R. Part 75: The uploader does not possess, and cannot produce, any valid government-issued age/identity verification or model release records for my depiction.
- EU Digital Services Act (Regulation (EU) 2022/2065) Article 16 Notice-and-Action mechanism for illegal content and non-consensual sharing of private material.
- GDPR Article 17 (Right to Erasure) and Article 9 (Unlawful processing of special-category biometric and sensitive personal data without explicit consent).

2. DMCA COPYRIGHT INFRINGEMENT (17 U.S.C. § 512(c)):
To the extent the media was self-captured or privately recorded on my device, I am also the exclusive copyright owner of the work. I have a good-faith belief that the use of this material is not authorized by me, any agent, or the law, and under penalty of perjury I affirm that the information in this notice is accurate.

3. DEMAND FOR IMMEDIATE REMOVAL & HASH BLACKLISTING:
I demand that you immediately:
a) Remove and disable public access to the content at {infringing_url}, including all video streams, preview thumbnails, storyboard sprites, and downloadable archives.
b) Purge the media from all edge CDNs, mirrors, and cache servers.
c) Compute and blacklist the cryptographic and perceptual hashes (MD5 / SHA-256 / PDQ / PhotoDNA) of the files to prevent automated or manual re-upload across your network.
d) Confirm completion of removal in writing to {subject_email}.

{f"Additional Forensic Details: {additional_notes}" if additional_notes else ""}

Signed,
{subject_name}
Contact Email: {subject_email}
"""
        return notice.strip()
