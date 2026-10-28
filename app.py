
from pathlib import Path

import joblib
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Banking Assistant",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# 2. LOAD YOUR TRAINED MODEL
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

tfidf = joblib.load(
    BASE_DIR / "model" / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    BASE_DIR / "model" / "intent_classifier.joblib"
)

# --------------------------------------------------
# 3. RESPONSE DICTIONARY
# --------------------------------------------------
responses = {
    "card_arrival": (
        "You're asking about your card's arrival. "
        "Please check your bank's app for delivery tracking."
    ),
    "card_delivery_estimate": (
        "You're asking about your card's delivery time. "
        "Check the estimated delivery date in your bank's app."
    ),
    "change_pin": (
        "You're asking how to change your PIN. "
        "Use your bank's official app or contact your bank."
    ),
    "why_verify_identity": (
        "You're asking why identity verification is needed. "
        "Check your bank's official guidance."
    ),
    "verify_my_identity": (
        "You're asking about identity verification. "
        "Follow your bank's official verification instructions."
    ),
}


# --------------------------------------------------
# 4. CHATBOT PREDICTION
# --------------------------------------------------
def chatbot_response(message):
    message_tfidf = tfidf.transform([message])
    intent = model.predict(message_tfidf)[0]

    if intent in responses:
        response = responses[intent]
    else:
        readable_intent = intent.replace("_", " ")
        response = (
            f"Your question appears to be about {readable_intent}. "
            "For accurate, account-specific help, please use "
            "your bank's official app or contact its support team."
        )

    return response, intent


# --------------------------------------------------
# 5. ANIMATED GLASSMORPHISM DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    
/* Restore the blue-purple gradient background */
.stApp {
    background:
        radial-gradient(
            ellipse at 15% 30%,
            rgba(0, 145, 255, 0.28),
            transparent 38%
        ),
        radial-gradient(
            ellipse at 85% 75%,
            rgba(130, 30, 255, 0.32),
            transparent 40%
        ),
        linear-gradient(
            135deg,
            #061329 0%,
            #0a102b 50%,
            #210b45 100%
        );
    background-attachment: fixed;
    min-height: 100vh;
}


f/* Chat bubble entrance animation */
[data-testid="stChatMessage"] {
    animation: chatBubbleEntrance 0.65s ease-out both;
}

/* Fade in and slide upward */
@keyframes chatBubbleEntrance {
    from {
        opacity: 0;
        transform: translateY(22px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Soft glow on hover */
[data-testid="stChatMessage"]:hover {
    box-shadow: 0 0 18px rgba(70, 150, 255, 0.22);
    transition: box-shadow 0.3s ease;
}

/* Respect reduced-motion settings */
@media (prefers-reduced-motion: reduce) {
    [data-testid="stChatMessage"] {
        animation: none !important;
    }
}

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# 6. HEADER WITH LOGO AND TAGLINE
# --------------------------------------------------
logo_path = BASE_DIR / "assets" / "banking_logo.png"

# Logo and branding layout
logo_column, title_column = st.columns(
    [1.25, 4.75],
    vertical_alignment="center",
)

with logo_column:
    # Glass-style logo container
    st.markdown(
        """
        <div class="logo-frame">
        """,
        unsafe_allow_html=True,
    )

    st.image(str(logo_path), width=170)

    st.markdown(
        """
        </div>
        <div class="logo-tagline">
            Most trusted bank by people
        </div>
        """,
        unsafe_allow_html=True,
    )

with title_column:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">
                YOUR DIGITAL BANKING COMPANION
            </div>
            <h1 class="hero-title">
                Banking Assistant
            </h1>
            <p class="hero-subtitle">
                A smarter way to explore banking questions.
                Ask about cards, PINs, verification, and more.
            </p>
            <div class="status-pill">
                ● AI intent classification demo
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# --------------------------------------------------
# 7. INITIALIZE CHAT HISTORY
# --------------------------------------------------
def welcome_message():
    return {
        "role": "assistant",
        "content": (
            "Hello! Welcome to your Banking Assistant. "
            "What would you like help with today?"
        ),
        "intent": None,
    }


if "messages" not in st.session_state:
    st.session_state.messages = [welcome_message()]


# --------------------------------------------------
# 8. CLEAR CHAT
# --------------------------------------------------
header_left, header_right = st.columns([5, 1])

with header_left:
    st.subheader("Your conversation")

with header_right:
    if st.button("Clear", use_container_width=True):
        st.session_state.messages = [welcome_message()]
        st.rerun()


# --------------------------------------------------
# 9. DISPLAY CHAT HISTORY
# --------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 10. SUGGESTED QUESTIONS
# --------------------------------------------------
st.markdown("#### Try asking")

suggestions = [
    "Where is my card?",
    "How do I change my PIN?",
    "Why do I need to verify my identity?",
]

columns = st.columns(3)

for index, suggestion in enumerate(suggestions):
    with columns[index]:
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = suggestion
            st.rerun()


# --------------------------------------------------
# 11. PROCESS USER MESSAGES
# --------------------------------------------------
def process_message(message):
    message = message.strip()

    if not message:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": message,
            "intent": None,
        }
    )

    response, intent = chatbot_response(message)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "intent": intent,
        }
    )


# Process a suggested question after rerun
if "pending_prompt" in st.session_state:
    pending_prompt = st.session_state.pop("pending_prompt")
    process_message(pending_prompt)
    st.rerun()


# Normal chat input
if prompt := st.chat_input("Type your banking question..."):
    process_message(prompt)
    st.rerun()


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------
st.divider()

st.markdown(
    """
    <p class="muted">
        Educational prototype trained on the Banking77 dataset.
        Not connected to a real bank and cannot access account details.
        Never enter passwords, PINs, OTPs, or card details.
    </p>
    """,
    unsafe_allow_html=True,
)
