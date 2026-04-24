"""
knowledge_base.py
-----------------
Department FAQ datasets for ShopUNow Assistant.
HR & Finance are internal-facing; Billing & Shipping are external (customer-facing).
"""

DATASETS = {
    "HR": [
        {"q": "How do I apply for paid time off (PTO)?", "a": "Submit a PTO request in Workday > Time Off. Manager approval required. Submit at least 3 business days in advance.", "department": "HR", "audience": "internal"},
        {"q": "What are ShopUNow's core working hours?", "a": "Core hours are 10:00–16:00 local time. Flexible start/end as agreed with your manager.", "department": "HR", "audience": "internal"},
        {"q": "Does ShopUNow offer parental leave?", "a": "Yes. 16 weeks paid primary caregiver leave and 6 weeks paid secondary caregiver leave. Coordinate with HR 30 days before expected start.", "department": "HR", "audience": "internal"},
        {"q": "How do I update my legal name?", "a": "Open an HR ticket with legal documentation. HR will update payroll, benefits, and directory within 5 business days.", "department": "HR", "audience": "internal"},
        {"q": "How can I access the employee handbook?", "a": "The handbook is in the HR Portal > Documents. Latest version is always pinned.", "department": "HR", "audience": "internal"},
        {"q": "Whom do I contact for workplace conflict mediation?", "a": "Create a confidential HR case in the HR Portal. An HRBP will reach out within 2 business days.", "department": "HR", "audience": "internal"},
        {"q": "Are there public holidays by region?", "a": "Yes. Holidays follow local calendars and are listed in the HR Portal > Regional Holidays.", "department": "HR", "audience": "internal"},
        {"q": "How do performance reviews work?", "a": "Biannual reviews via Workday: self-review, peer/manager feedback, calibration, final rating.", "department": "HR", "audience": "internal"},
        {"q": "Can I work remotely?", "a": "Remote/hybrid depends on role and manager approval. See Remote Work Policy in HR Portal.", "department": "HR", "audience": "internal"},
        {"q": "Where do I report harassment?", "a": "Use the confidential Ethics & Compliance form or contact HRBP immediately.", "department": "HR", "audience": "internal"},
        {"q": "How do I enroll in benefits?", "a": "Enroll during onboarding or open enrollment via Benefits Center in Workday.", "department": "HR", "audience": "internal"},
        {"q": "What training is mandatory?", "a": "Security, Code of Conduct, and Anti-harassment trainings annually via LMS.", "department": "HR", "audience": "internal"},
    ],
    "Finance": [
        {"q": "What is the expense reimbursement policy?", "a": "Submit expenses within 30 days via Concur. Receipts required for items > $25. Travel must be pre-approved.", "department": "Finance", "audience": "internal"},
        {"q": "How long do reimbursements take?", "a": "Approved claims are paid in the next weekly AP run (typically 5–7 business days).", "department": "Finance", "audience": "internal"},
        {"q": "Can I buy software with my card?", "a": "All software requires vendor security review and Finance approval before purchase.", "department": "Finance", "audience": "internal"},
        {"q": "What's the fiscal year?", "a": "ShopUNow fiscal year runs Jan 1 – Dec 31.", "department": "Finance", "audience": "internal"},
        {"q": "How are per diems handled?", "a": "Per diems follow government/region tables; claim via Concur with travel dates and destination.", "department": "Finance", "audience": "internal"},
        {"q": "Where do I find cost centers?", "a": "Cost center list is in the Finance Wiki. Ask your manager for your team's code.", "department": "Finance", "audience": "internal"},
        {"q": "How do vendor invoices get paid?", "a": "Vendors email invoices to ap@shopunow.com with PO. Net-30 unless negotiated.", "department": "Finance", "audience": "internal"},
        {"q": "Who approves capital expenses?", "a": "Department head and Finance Controller must approve capex > $5,000.", "department": "Finance", "audience": "internal"},
        {"q": "What's the company travel card policy?", "a": "Issued to frequent travelers. Card use limited to travel-related expenses.", "department": "Finance", "audience": "internal"},
        {"q": "Exchange rate for international claims?", "a": "Use Concur's daily rate on the expense date unless a receipt shows the charged currency.", "department": "Finance", "audience": "internal"},
        {"q": "How do I request a PO?", "a": "Submit a PR in ProcureNow with vendor, quote, and cost center. PO created after approvals.", "department": "Finance", "audience": "internal"},
        {"q": "Who to contact for payroll issues?", "a": "Open a Payroll ticket in the Finance Portal. Response in 1–2 business days.", "department": "Finance", "audience": "internal"},
    ],
    "Billing": [
        {"q": "Why was my card declined?", "a": "Common reasons include insufficient funds or bank security checks. Try another card or contact your bank; you can also use PayPal.", "department": "Billing", "audience": "external"},
        {"q": "How do I download my invoice?", "a": "Go to Account > Orders, select the order, then click Download Invoice.", "department": "Billing", "audience": "external"},
        {"q": "Can I change the billing address after purchase?", "a": "Yes, within 30 minutes of checkout from Order Details > Edit Billing Address.", "department": "Billing", "audience": "external"},
        {"q": "Do you accept BNPL?", "a": "Yes, we support ShopPay Installments and Klarna where available.", "department": "Billing", "audience": "external"},
        {"q": "Why do I see two charges?", "a": "You may see a temporary authorization hold and the final charge. Holds drop off in 3–5 business days.", "department": "Billing", "audience": "external"},
        {"q": "How do I apply a promo code?", "a": "Enter the code at checkout in the Promo field before payment.", "department": "Billing", "audience": "external"},
        {"q": "Can I split payment methods?", "a": "Yes, you can split between one card and a gift card/store credit.", "department": "Billing", "audience": "external"},
        {"q": "Refund timeline?", "a": "Once approved, refunds post to your bank in 5–10 business days.", "department": "Billing", "audience": "external"},
        {"q": "Currency support?", "a": "We charge in your local currency where supported; otherwise in USD with your bank's conversion.", "department": "Billing", "audience": "external"},
        {"q": "VAT invoice available?", "a": "Yes. Add your tax ID at checkout; invoice will include VAT details.", "department": "Billing", "audience": "external"},
        {"q": "How to update saved cards?", "a": "Account > Payment Methods to add/remove cards securely.", "department": "Billing", "audience": "external"},
        {"q": "My promo code isn't working.", "a": "Check expiration, minimum spend, or category exclusions; contact support if issues persist.", "department": "Billing", "audience": "external"},
    ],
    "Shipping": [
        {"q": "When will my order ship?", "a": "Orders ship within 1–2 business days. You'll receive a tracking email once dispatched.", "department": "Shipping", "audience": "external"},
        {"q": "How do I track my package?", "a": "Use the tracking link in your email or go to Account > Orders and click Track.", "department": "Shipping", "audience": "external"},
        {"q": "Do you offer expedited shipping?", "a": "Yes: Standard, Expedited (2-day), and Priority (next business day) at checkout.", "department": "Shipping", "audience": "external"},
        {"q": "What if my package is late?", "a": "If tracking hasn't updated for 3 business days, contact support to open a carrier trace.", "department": "Shipping", "audience": "external"},
        {"q": "International shipping fees?", "a": "Shown at checkout. Customs/duties may apply and are displayed when available.", "department": "Shipping", "audience": "external"},
        {"q": "Can I change the address after ordering?", "a": "Edits are possible within 30 minutes from Order Details > Edit Shipping Address.", "department": "Shipping", "audience": "external"},
        {"q": "My tracking shows delivered but I didn't receive it.", "a": "Check neighbors and safe places. If not found in 24 hours, contact support for a lost-package claim.", "department": "Shipping", "audience": "external"},
        {"q": "Do you ship to PO boxes?", "a": "Yes for Standard shipping, not for Priority.", "department": "Shipping", "audience": "external"},
        {"q": "What is the return window?", "a": "30 days from delivery for most items. Start a return from Account > Returns.", "department": "Shipping", "audience": "external"},
        {"q": "How are fragile items packed?", "a": "We use protective materials and 'Fragile' labels; report damage with photos within 7 days.", "department": "Shipping", "audience": "external"},
        {"q": "Can I hold a shipment?", "a": "After dispatch, some carriers support Hold at Location; use your tracking link to request.", "department": "Shipping", "audience": "external"},
        {"q": "What are free-shipping thresholds?", "a": "Free Standard shipping on domestic orders over $50 (after discounts).", "department": "Shipping", "audience": "external"},
    ],
}

DEPARTMENTS = list(DATASETS.keys())
